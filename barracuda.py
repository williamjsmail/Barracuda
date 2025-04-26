from flask import Flask, render_template, request, jsonify, Response
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
from docx import Document
from urllib.parse import quote


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Global cache for model and precomputed data
MODEL_CACHE = {
    'model': None,
    'tech_embeddings': None,
    'tech_ids': None,
    'key_terms': None
}


def load_techniques_enriched(mitre_json_path):
    with open(mitre_json_path, "r") as f:
        data = json.load(f)
    
    # parse all techniques and their names
    technique_names = {}
    for obj in data.get("objects", []):
        if obj.get("type") == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            tech_id = next((ref["external_id"] for ref in ext_refs if ref["source_name"] == "mitre-attack"), "")
            name = obj.get("name", "")
            if tech_id and name:
                technique_names[tech_id] = name
    
    # build techniques with full names
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
                
                # determine if it's a sub-technique
                is_subtechnique = obj.get("x_mitre_is_subtechnique", False)
                if is_subtechnique:
                    parent_id = tech_id.split(".")[0]
                    parent_name = technique_names.get(parent_id, "Unknown")
                    full_name = f"{parent_name}: {name}"
                else:
                    full_name = name
                
                techniques[tech_id] = {"text": combined_text, "name": full_name}
    
    return techniques


# Load techniques at startup
techniques = load_techniques_enriched("enterprise-attack.json")


def initialize_cache(techniques, batch_size=32):
    # Load model
    model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Precompute technique embeddings
    tech_ids = np.array(list(techniques.keys()))
    tech_texts = [t["text"] for t in techniques.values()]
    tech_embeddings = model.encode(
        tech_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    
    # Compute TF-IDF (term freq interdoc freq) key terms
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(tech_texts)
    feature_names = vectorizer.get_feature_names_out()
    key_terms = {}
    for i, tech_id in enumerate(techniques.keys()):
        tfidf_vector = tfidf_matrix[i].toarray()[0]
        top_indices = tfidf_vector.argsort()[-5:][::-1]
        key_terms[tech_id] = [(feature_names[idx], tfidf_vector[idx]) for idx in top_indices]
    
    # Store in cache
    MODEL_CACHE['model'] = model
    MODEL_CACHE['tech_embeddings'] = tech_embeddings
    MODEL_CACHE['tech_ids'] = tech_ids
    MODEL_CACHE['key_terms'] = key_terms


# Initialize cache at startup
initialize_cache(techniques)


def fetch_report_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from specific tags, prioritizing meaningful content
        content_tags = soup.find_all(['p', 'article', 'section'])
        seen_text = set()
        text_parts = []
        for tag in content_tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 20 and text not in seen_text:
                text_parts.append(text)
                seen_text.add(text)
        
        # Join and clean the text
        full_text = ' '.join(text_parts)
        full_text = re.sub(r'\s+', ' ', full_text.strip())
        
        if not full_text:
            raise ValueError("No meaningful text extracted from the URL.")
        
        return full_text
    
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""
    except ValueError as e:
        print(f"Error processing content: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""


def extract_matches_with_boosts(text, techniques, top_n=5, batch_size=32, min_score=0.3):
    # Retrieve cached data
    model = MODEL_CACHE['model']
    tech_embeddings = MODEL_CACHE['tech_embeddings']
    tech_ids = MODEL_CACHE['tech_ids']
    key_terms = MODEL_CACHE['key_terms']
    
    # Removing duplicates
    sentences = [
        s.strip() for s in re.split(r'[.!?]+', text)
        if len(s.strip()) > 20 and not s.strip().startswith(('http', 'www'))
    ]
    unique_sentences = list(dict.fromkeys(sentences))  # Deduplicate sentences
    if not unique_sentences:
        return [], {}
    
    # Encode sentences
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
            
            # Boost based on key term matches
            matched_scores = [score for term, score in key_terms[tid] if term in sentence_words]
            boost = sum(matched_scores)
            boosted_score = base_score + boost
            
            # Penalize short sentences
            if len(sentence.split()) < 15 and boosted_score < 0.7:
                boosted_score *= 0.9
                
            results.append((tid, sentence, min(boosted_score, 1.0)))
        return results
    
    # Parallel processing
    results = []
    with ThreadPoolExecutor() as executor:
        for batch in executor.map(process_sentence, range(len(unique_sentences))):
            results.extend(batch)
    
    # Deduplicate results, keeping highest score for each sentence-technique pair
    seen_pairs = {}
    for tcode, sentence, score in results:
        key = (sentence, tcode)
        if key not in seen_pairs or score > seen_pairs[key][2]:
            seen_pairs[key] = (tcode, sentence, score)
    
    # Group techniques by sentence
    sentence_to_techniques = defaultdict(list)
    for tcode, sentence, score in seen_pairs.values():
        sentence_to_techniques[sentence].append((tcode, score))
    
    # Aggregate results for occurrence-based boost
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
    
    # Dynamic threshold
    if adjusted_results:
        max_score = max(max(tech[1] for tech in techs) for _, techs in adjusted_results)
        threshold = max(0.5, max_score * 0.7)
    else:
        threshold = 0.5
    
    # Filter and sort results, and populate tcode_counts
    final_results = []
    for sentence, tech_list in adjusted_results:
        filtered_techs = [(tcode, score) for tcode, score in tech_list if score >= threshold]
        if filtered_techs:
            final_results.append((sentence, filtered_techs))
            for tcode, _ in filtered_techs:
                tcode_counts[tcode] += 1
    
    # Sort by highest technique score
    final_results.sort(key=lambda x: max(score for _, score in x[1]), reverse=True)
    
    return final_results, tcode_counts


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            if file.filename.lower().endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
            elif file.filename.lower().endswith('.docx'):
                content = extract_text_from_docx(file_path)
            else:
                return jsonify({'content': 'Unsupported file type. Please upload a PDF or DOCX file.'})
            if content:
                final_results, tcode_counts = extract_matches_with_boosts(content, techniques)
                tcode_to_sentences = defaultdict(list)
                for sentence, tech_list in final_results:
                    for tcode, score in tech_list:
                        tcode_to_sentences[tcode].append({"sentence": sentence, "score": score})
                tcode_names = {tcode: tech["name"] for tcode, tech in techniques.items() if tcode in tcode_counts}
                return jsonify({
                    'full_text': content,
                    'tcode_to_sentences': dict(tcode_to_sentences),
                    'tcode_counts': dict(tcode_counts),
                    'tcode_names': tcode_names,
                    'source_type': 'file'
                })
            else:
                return jsonify({'content': 'Failed to extract text from the uploaded file. The file may be corrupted or contain no extractable text (e.g., scanned images).'})
    elif 'url' in request.form:
        url = request.form['url']
        content = fetch_report_text(url)
        if content:
            final_results, tcode_counts = extract_matches_with_boosts(content, techniques)
            tcode_to_sentences = defaultdict(list)
            for sentence, tech_list in final_results:
                for tcode, score in tech_list:
                    tcode_to_sentences[tcode].append({"sentence": sentence, "score": score})
            tcode_names = {tcode: tech["name"] for tcode, tech in techniques.items() if tcode in tcode_counts}
            return jsonify({
                'full_text': content,
                'tcode_to_sentences': dict(tcode_to_sentences),
                'tcode_counts': dict(tcode_counts),
                'tcode_names': tcode_names,
                'source_type': 'url',
                'url': url
            })
        else:
            return jsonify({'content': 'Failed to fetch or extract text from the URL.'})
    return jsonify({'content': 'No file or URL provided.'})


@app.route('/proxy', methods=['GET'])
def proxy_webpage():
    url = request.args.get('url')
    sentences = request.args.get('sentences', '').split('||')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Create JavaScript for highlighting
        highlight_script = f"""
        <script>
            window.addEventListener('load', function() {{
                const sentences = {json.dumps(sentences)};
                const textNodes = document.createTreeWalker(
                    document.body,
                    Node.TEXT_NODE,
                    null,
                    false
                );
                let node;
                while (node = textNodes.nextNode()) {{
                    let text = node.nodeValue;
                    sentences.forEach(sentence => {{
                        const escapedSentence = sentence.replace(/[-[\]{{}}()*+?.,\\^$|#]/g, '\\\\$&');
                        const regex = new RegExp('(' + escapedSentence + ')', 'gi');
                        if (regex.test(text)) {{
                            const span = document.createElement('span');
                            span.style.backgroundColor = 'yellow';
                            span.innerHTML = text.replace(regex, '<span style="background-color: yellow;">$1</span>');
                            node.parentNode.replaceChild(span, node);
                        }}
                    }});
                }}
            }});
        </script>
        """
        
        # Inject script into the head
        if soup.head:
            soup.head.append(BeautifulSoup(highlight_script, 'html.parser'))
        else:
            soup.insert(0, BeautifulSoup('<head>' + highlight_script + '</head>', 'html.parser'))
        
        # Return modified HTML
        return Response(str(soup), mimetype='text/html')
    
    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch webpage: {e}"}), 500
    except Exception as e:
        return jsonify({'error': f"Error processing webpage: {e}"}), 500


def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + ' '
            text = re.sub(r'\s+', ' ', text.strip())
            return text if text else ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = '\n'.join(para.text for para in doc.paragraphs)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False)
