from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import os
import pdfplumber
import logging
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("pytesseract not installed; OCR for scanned PDFs unavailable")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/serve_pdf/*": {"origins": "*"}})
UPLOAD_FOLDER = 'Uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global cache for model and precomputed data
MODEL_CACHE = {
    'model': None,
    'tech_embeddings': None,
    'tech_ids': None,
    'key_terms': None
}

def normalize_sentence(sentence):
    if not sentence:
        return ""
    return re.sub(r'[\'\"\\]', '', re.sub(r'\s+', ' ', sentence.strip()).replace('[\u200B-\u200D\uFEFF]', '').replace('[\x00-\x1F\x7F]', ''))

def load_techniques_enriched(mitre_json_path):
    try:
        with open(mitre_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load MITRE JSON: {e}")
        raise
    
    technique_names = {}
    for obj in data.get("objects", []):
        if obj.get("type") == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            tech_id = next((ref["external_id"] for ref in ext_refs if ref["source_name"] == "mitre-attack"), "")
            name = obj.get("name", "")
            if tech_id and name:
                technique_names[tech_id] = name
    
    techniques = {}
    for obj in data.get("objects", []):
        if obj.get("type") == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            tech_id = next((ref["external_id"] for ref in ext_refs if ref["source_name"] == "mitre-attack"), "")
            name = obj.get("name", "")
            desc = obj.get("description", "")
            if tech_id and name and desc:
                desc = re.sub(r'\s+', ' ', desc.strip())
                combined_text = f"{tech_id} {name}. {desc}"
                
                is_subtechnique = obj.get("x_mitre_is_subtechnique", False)
                if is_subtechnique:
                    parent_id = tech_id.split(".")[0]
                    parent_name = technique_names.get(parent_id, "Unknown")
                    full_name = f"{parent_name}: {name}"
                else:
                    full_name = name
                
                techniques[tech_id] = {"text": combined_text, "name": full_name}
    
    return techniques

try:
    techniques = load_techniques_enriched("enterprise-attack.json")
except Exception as e:
    logger.error(f"Failed to initialize techniques: {e}")
    raise

def initialize_cache(techniques, batch_size=32):
    try:
        model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        tech_ids = np.array(list(techniques.keys()))
        tech_texts = [t["text"] for t in techniques.values()]
        tech_embeddings = model.encode(
            tech_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        tfidf_matrix = vectorizer.fit_transform(tech_texts)
        feature_names = vectorizer.get_feature_names_out()
        key_terms = {}
        for i, tech_id in enumerate(techniques.keys()):
            tfidf_vector = tfidf_matrix[i].toarray()[0]
            top_indices = tfidf_vector.argsort()[-5:][::-1]
            key_terms[tech_id] = [(feature_names[idx], tfidf_vector[idx]) for idx in top_indices]
        
        MODEL_CACHE['model'] = model
        MODEL_CACHE['tech_embeddings'] = tech_embeddings
        MODEL_CACHE['tech_ids'] = tech_ids
        MODEL_CACHE['key_terms'] = key_terms
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        raise

try:
    initialize_cache(techniques)
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    raise

def fetch_report_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        content_tags = soup.find_all(['p', 'article', 'section', 'span', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        seen_text = set()
        text_parts = []
        for tag in content_tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 20 and text not in seen_text:
                text_parts.append(text)
                seen_text.add(text)
        
        full_text = ' '.join(text_parts)
        full_text = normalize_sentence(full_text)
        
        if not full_text:
            raise ValueError("No meaningful text extracted from the URL.")
        
        return full_text
    
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return ""
    except ValueError as e:
        logger.error(f"Error processing content from {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return ""

def extract_matches_with_boosts(text, techniques, top_n=5, batch_size=32, min_score=0.3):
    try:
        model = MODEL_CACHE['model']
        tech_embeddings = MODEL_CACHE['tech_embeddings']
        tech_ids = MODEL_CACHE['tech_ids']
        key_terms = MODEL_CACHE['key_terms']
        
        sentences = [
            normalize_sentence(s) for s in re.split(r'[.!?]+', text)
            if len(s.strip()) > 20 and not s.strip().startswith(('http', 'www'))
        ]
        unique_sentences = list(dict.fromkeys(sentences))
        logger.info(f"Extracted {len(unique_sentences)} unique sentences for matching")
        for idx, s in enumerate(unique_sentences[:5]):  # Log first 5 for debugging
            logger.info(f"Sample sentence {idx + 1}: {s}")
        
        if not unique_sentences:
            return [], {}
        
        sentence_embeddings = model.encode(
            unique_sentences,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        def process_sentence(i):
            sims = util.cos_sim(sentence_embeddings[i], tech_embeddings)[0]
            top_indices = np.argsort(sims.cpu().numpy())[::-1][:top_n]
            sentence = unique_sentences[i]
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            
            results = []
            for idx in top_indices:
                tid = tech_ids[idx]
                base_score = sims[idx].item()
                if base_score < min_score:
                    continue
                
                matched_scores = [score for term, score in key_terms[tid] if term in sentence_words]
                boost = sum(matched_scores)
                boosted_score = base_score + boost
                
                if len(sentence.split()) < 15 and boosted_score < 0.7:
                    boosted_score *= 0.9
                    
                results.append((tid, sentence, min(boosted_score, 1.0)))
            return results
        
        results = []
        with ThreadPoolExecutor() as executor:
            for batch in executor.map(process_sentence, range(len(unique_sentences))):
                results.extend(batch)
        
        seen_pairs = {}
        for tcode, sentence, score in results:
            key = (sentence, tcode)
            if key not in seen_pairs or score > seen_pairs[key][2]:
                seen_pairs[key] = (tcode, sentence, score)
        
        sentence_to_techniques = defaultdict(list)
        for tcode, sentence, score in seen_pairs.values():
            sentence_to_techniques[sentence].append((tcode, score))
        
        BOOST_PER_EXTRA_OCCURRENCE = 0.03
        adjusted_results = []
        tcode_counts = defaultdict(int)
        for sentence, tech_list in sentence_to_techniques.items():
            boosted_techs = []
            for tcode, score in tech_list:
                temp_count = sum(1 for _, tlist in sentence_to_techniques.items() if any(tc == tcode for tc, _ in tlist))
                boosted_score = score + ((temp_count - 1) * BOOST_PER_EXTRA_OCCURRENCE) if temp_count > 1 else score
                boosted_techs.append((tcode, min(boosted_score, 1.0)))
            boosted_techs.sort(key=lambda x: x[1], reverse=True)
            adjusted_results.append((sentence, boosted_techs))
        
        if adjusted_results:
            max_score = max(max(tech[1] for tech in techs) for _, techs in adjusted_results)
            threshold = max(0.5, max_score * 0.7)
        else:
            threshold = 0.5
        
        final_results = []
        for sentence, tech_list in adjusted_results:
            filtered_techs = [(tcode, score) for tcode, score in tech_list if score >= threshold]
            if filtered_techs:
                final_results.append((sentence, filtered_techs))
                for tcode, _ in filtered_techs:
                    tcode_counts[tcode] += 1
        
        final_results.sort(key=lambda x: max(score for _, score in x[1]), reverse=True)
        
        return final_results, tcode_counts
    
    except Exception as e:
        logger.error(f"Error extracting matches: {e}")
        return [], {}

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Handle JSON payload for raw text
        if request.content_type == 'application/json':
            logger.info("Received JSON request for raw text")
            data = request.get_json()
            if not data or 'text' not in data:
                logger.warning("No text provided in JSON request")
                return jsonify({'error': 'No text provided'}), 400
            text = data['text']
            if not text.strip():
                logger.warning("Empty text provided in JSON request")
                return jsonify({'error': 'Text cannot be empty'}), 400            
            logger.info("Processing raw text")
            content = normalize_sentence(text)
            if content:
                final_results, tcode_counts = extract_matches_with_boosts(content, techniques)
                tcode_to_sentences = defaultdict(list)
                for sentence, tech_list in final_results:
                    for tcode, score in tech_list:
                        tcode_to_sentences[tcode].append({"sentence": sentence, "score": score})
                tcode_names = {tcode: tech["name"] for tcode, tech in techniques.items() if tcode in tcode_counts}
                logger.info("Successfully processed raw text")
                return jsonify({
                    'full_text': content,
                    'tcode_to_sentences': dict(tcode_to_sentences),
                    'tcode_counts': dict(tcode_counts),
                    'tcode_names': tcode_names,
                    'source_type': 'text'
                })
            else:
                logger.error("Failed to process raw text: no content extracted")
                return jsonify({'content': 'Failed to process raw text.'}), 400
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file_path = os.path.normpath(file_path)
            logger.info(f"Saving file to {file_path}")
            file.save(file_path)
            if not os.path.exists(file_path):
                logger.error(f"File not saved: {file_path}")
                return jsonify({'content': 'Failed to save file.'}), 500
            if file.filename.lower().endswith('.pdf'):
                logger.info(f"Extracting text from PDF: {file.filename}")
                content = extract_text_from_pdf(file_path)
                if content:
                    final_results, tcode_counts = extract_matches_with_boosts(content, techniques)
                    tcode_to_sentences = defaultdict(list)
                    for sentence, tech_list in final_results:
                        for tcode, score in tech_list:
                            tcode_to_sentences[tcode].append({"sentence": sentence, "score": score})
                    tcode_names = {tcode: tech["name"] for tcode, tech in techniques.items() if tcode in tcode_counts}
                    logger.info(f"Successfully processed PDF: {file.filename}")
                    return jsonify({
                        'full_text': content,
                        'tcode_to_sentences': dict(tcode_to_sentences),
                        'tcode_counts': dict(tcode_counts),
                        'tcode_names': tcode_names,
                        'source_type': 'file',
                        'pdf_filename': file.filename
                    })
                else:
                    error_msg = 'Failed to extract text from the PDF.'
                    if OCR_AVAILABLE:
                        error_msg += ' Try enabling OCR for scanned PDFs.'
                    logger.error(f"{error_msg}: {file.filename}")
                    return jsonify({'content': error_msg}), 400
            else:
                logger.warning(f"Unsupported file type: {file.filename}")
                return jsonify({'content': 'Unsupported file type. Please upload a PDF file.'}), 400
        elif 'url' in request.form and request.form['url']:
            url = request.form['url']
            logger.info(f"Fetching URL: {url}")
            content = fetch_report_text(url)
            if content:
                final_results, tcode_counts = extract_matches_with_boosts(content, techniques)
                tcode_to_sentences = defaultdict(list)
                for sentence, tech_list in final_results:
                    for tcode, score in tech_list:
                        tcode_to_sentences[tcode].append({"sentence": sentence, "score": score})
                tcode_names = {tcode: tech["name"] for tcode, tech in techniques.items() if tcode in tcode_counts}
                logger.info(f"Successfully processed URL: {url}")
                return jsonify({
                    'full_text': content,
                    'tcode_to_sentences': dict(tcode_to_sentences),
                    'tcode_counts': dict(tcode_counts),
                    'tcode_names': tcode_names,
                    'source_type': 'url',
                    'url': url
                })
            else:
                logger.error(f"Failed to fetch or extract text from URL: {url}")
                return jsonify({'content': 'Failed to fetch or extract text from the URL.'}), 400
        logger.warning("No file or URL provided in upload request")
        return jsonify({'content': 'No file or URL provided.'}), 400
    
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

@app.route('/serve_pdf/<filename>')
def serve_pdf(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_path = os.path.normpath(file_path)
        logger.info(f"Attempting to serve PDF: {filename}, path: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return jsonify({'error': 'PDF file not found'}), 404
        logger.info(f"Serving PDF: {filename}")
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            mimetype='application/pdf',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"File access error for {file_path}: {e}")
        return jsonify({'error': f"Failed to serve PDF: {e}"}), 500

def extract_text_from_pdf(file_path):
    try:
        logger.info(f"Opening PDF: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + ' '
                    logger.info(f"Extracted text from page {page_num} of {file_path}")
                else:
                    logger.warning(f"No text extracted from page {page_num} of {file_path}")
                    if OCR_AVAILABLE:
                        logger.info(f"Attempting OCR on page {page_num}")
                        try:
                            image = page.to_image(resolution=300)
                            page_text = pytesseract.image_to_string(image.original)
                            if page_text:
                                text += page_text + ' '
                                logger.info(f"OCR extracted text from page {page_num}")
                            else:
                                logger.warning(f"OCR found no text on page {page_num}")
                        except Exception as ocr_e:
                            logger.error(f"OCR failed on page {page_num}: {ocr_e}")
            text = normalize_sentence(text)
            if not text:
                logger.error(f"No text extracted from PDF: {file_path}")
            return text if text else ""
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        logger.info(f"Creating uploads folder: {UPLOAD_FOLDER}")
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False, host='0.0.0.0', port=5000)
