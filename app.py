from flask import Flask, render_template, request, jsonify, send_file
import os
import re
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import json
import csv
from io import StringIO, BytesIO
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['SUBMISSIONS_FOLDER'] = os.path.join(os.getcwd(), 'submissions')
app.config['APPROVED_FOLDER'] = os.path.join(os.getcwd(), 'approved_papers')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SUBMISSIONS_FOLDER'], exist_ok=True)
os.makedirs(app.config['APPROVED_FOLDER'], exist_ok=True)

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(pdf_path):
    text = ""
    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("pdfplumber error:", e)

    # If no text found, fallback to OCR
    if not text.strip():
        try:
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            print("OCR error:", e)
    return text.strip()

# ---------------- PUBLISHER / FORMAT DETECTION ----------------
def detect_format(text):
    text_lower = text.lower()

    # DOI detection (existing logic)
    doi_match = re.search(r"10\.\d{4,9}/\S+", text.replace("\n", ""))
    if doi_match:
        doi = doi_match.group(0)
        if doi.startswith("10.1109") or "ieee" in text_lower:
            return "IEEE Format", 90
        elif doi.startswith("10.1007") or "springer" in text_lower:
            return "Springer Format", 90
        elif doi.startswith("10.1016") or "elsevier" in text_lower:
            return "Elsevier Format", 90
        elif doi.startswith("10.1145") or "acm" in text_lower:
            return "ACM Format", 90

    # Expanded keyword detection
    if "ieee" in text_lower or "digital object identifier" in text_lower:
        return "IEEE Format", 80
    elif "springer" in text_lower or "springer nature" in text_lower:
        return "Springer Format", 80
    elif "elsevier" in text_lower or "journal homepage" in text_lower:
        return "Elsevier Format", 80
    elif "acm" in text_lower or "association for computing machinery" in text_lower:
        return "ACM Format", 80
    elif "engineering and science" in text_lower or "e&s" in text_lower:
        return "E&S Format", 70
    elif "physicae organum" in text_lower or "universidade de brasília" in text_lower:
        # Detect Physicae Organum
        return "Physicae Organum Format", 85
    else:
        return "Unknown / Custom Format", 0

# ---------------- ENHANCED AI DETECTION ----------------
def detect_ai_content(text):
    """
    Enhanced AI detection with better accuracy for published papers
    """
    if not text or len(text) < 100:
        return {"ai_score": 0, "human_score": 100, "confidence": "Low", "method": "Insufficient Text"}
    
    # Enhanced heuristic detection
    heuristic_score = enhanced_heuristic_detection(text)
    
    # Published paper adjustment
    published_adjustment = check_published_paper_indicators(text)
    
    # Apply adjustment (reduce AI score for published papers)
    adjusted_score = max(0, heuristic_score - published_adjustment)
    
    return {
        "ai_score": round(adjusted_score, 2),
        "human_score": round(100 - adjusted_score, 2),
        "confidence": get_confidence_level(adjusted_score),
        "method": "Enhanced Heuristic Analysis",
        "published_indicators": published_adjustment > 0
    }

def enhanced_heuristic_detection(text):
    """
    Enhanced heuristic detection with better pattern recognition
    """
    score = 0
    text_lower = text.lower()
    words = text_lower.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    # 1. Check for obvious AI indicators
    ai_indicators = [
        (r"\b(as an ai|as a language model|i am an ai|i cannot|as an assistant)\b", 25),
        (r"\b(in conclusion|to summarize|overall|in summary)\b", 3),
        (r"\b(it is important to note|it is worth mentioning|it should be noted)\b", 4),
        (r"\b(however|furthermore|moreover|additionally|consequently)\b", 2),
    ]
    
    for pattern, points in ai_indicators:
        matches = len(re.findall(pattern, text_lower))
        score += min(matches * points, 15)
    
    # 2. Analyze sentence structure variety
    if len(sentences) > 10:
        # Check for sentence length variety
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        std_length = (sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)) ** 0.5
        
        # Low variety in sentence length might indicate AI
        if std_length < 4:
            score += 10
        elif std_length < 6:
            score += 5
    
    # 3. Check for repetitive phrases
    if len(words) > 200:
        word_freq = {}
        for word in words:
            if len(word) > 5:  # Only consider meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate repetition score
        repetitive_words = sum(1 for count in word_freq.values() if count > 5)
        repetition_ratio = repetitive_words / len(word_freq) if word_freq else 0
        score += min(repetition_ratio * 50, 20)
    
    # 4. Check for academic writing patterns (reduce false positives)
    academic_indicators = [
        'references', 'bibliography', 'citation', 'cited', 'et al', 'figure',
        'table', 'methodology', 'results', 'discussion', 'abstract', 'introduction'
    ]
    
    academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
    if academic_score >= 3:
        score -= 10  # Reduce AI score for academic papers
    
    # 5. Check paragraph structure
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 5:
        avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
        if 50 <= avg_para_length <= 200:  # Typical academic paragraph length
            score -= 5
    
    return max(0, min(score, 80))

def check_published_paper_indicators(text):
    """
    Check for indicators that this is a published academic paper
    """
    text_lower = text.lower()
    adjustment = 0
    
    # Strong published paper indicators
    strong_indicators = [
        r'\bdoi:\s*\d{2}\.\d{4,9}/[-._;()/:a-z0-9]+\b',
        r'\bissn:\s*\d{4}-\d{3}[\dxX]\b',
        r'published in',
        r'journal of',
        r'proceedings of',
        r'vol\.\s*\d+',
        r'no\.\s*\d+',
        r'pp\.\s*\d+-\d+',
        r'received.*accepted',
        r'copyright\s*©',
        r'all rights reserved',
        r'peer.reviewed',
    ]
    
    strong_matches = sum(1 for pattern in strong_indicators if re.search(pattern, text_lower))
    if strong_matches >= 3:
        adjustment += 25
    
    # Medium indicators
    medium_indicators = [
        'references', 'bibliography', 'citation', 'abstract', 'keywords',
        'introduction', 'methodology', 'results', 'discussion', 'conclusion',
        'acknowledgements', 'corresponding author'
    ]
    
    medium_matches = sum(1 for indicator in medium_indicators if indicator in text_lower)
    adjustment += min(medium_matches * 2, 15)
    
    # Check for formal academic structure
    sections = ['abstract', 'introduction', 'method', 'results', 'discussion', 'conclusion']
    section_matches = sum(1 for section in sections if re.search(rf'\b{section}\b', text_lower))
    if section_matches >= 4:
        adjustment += 10
    
    return min(adjustment, 40)  # Cap adjustment at 40%

def get_confidence_level(score):
    """Determine confidence level based on AI score"""
    if score < 20 or score > 80:
        return "High"
    elif score < 40 or score > 60:
        return "Medium"
    else:
        return "Low"

# ---------------- SIMILARITY CALCULATION ----------------
def compute_similarity(texts):
    if len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

# ---------------- SUBMISSION MANAGEMENT ----------------
def save_submission_data(filenames, results, similarity_data, ai_results):
    """Save submission data to track all analyses"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"submission_{timestamp}"
    
    submission_data = {
        'submission_id': submission_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'files': filenames,
        'results': results,
        'similarity': similarity_data,
        'ai_analysis': ai_results,
        'submission_status': 'analyzed'
    }
    
    # Save as JSON
    json_path = os.path.join(app.config['SUBMISSIONS_FOLDER'], f"{submission_id}.json")
    with open(json_path, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    return submission_id

def submit_paper(filename, file_path, analysis_result, ai_result):
    """Submit paper for approval if it meets criteria"""
    ai_score = ai_result.get('ai_score', 0)
    
    submission_data = {
        'filename': filename,
        'submission_time': datetime.datetime.now().isoformat(),
        'ai_score': ai_score,
        'human_score': ai_result.get('human_score', 0),
        'format': analysis_result['format'],
        'confidence': analysis_result['confidence'],
        'status': 'approved' if ai_score <= 25 else 'review_required',
        'published_indicators': ai_result.get('published_indicators', False)
    }
    
    # Save to approved folder if AI score is low enough
    if ai_score <= 25:
        approved_path = os.path.join(app.config['APPROVED_FOLDER'], filename)
        with open(approved_path, 'wb') as f:
            with open(file_path, 'rb') as source_file:
                f.write(source_file.read())
        submission_data['status'] = 'automatically_approved'
    
    # Save submission record
    submission_id = f"submit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    json_path = os.path.join(app.config['SUBMISSIONS_FOLDER'], f"{submission_id}.json")
    with open(json_path, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    return submission_data

# ---------------- REPORT GENERATION ----------------
def generate_csv_report(results, similarity_data, ai_results):
    """Generate CSV report"""
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['PDF Analysis Report', 'Generated on', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    
    # Individual File Analysis
    writer.writerow(['FILE ANALYSIS'])
    writer.writerow(['Filename', 'Format', 'Confidence', 'AI Score', 'Human Score', 'AI Confidence', 'Published Indicators'])
    
    for result in results:
        filename = result['filename']
        ai_info = ai_results.get(filename, {})
        writer.writerow([
            filename,
            result['format'],
            result['confidence'],
            f"{ai_info.get('ai_score', 0)}%",
            f"{ai_info.get('human_score', 100)}%",
            ai_info.get('confidence', 'Unknown'),
            "Yes" if ai_info.get('published_indicators', False) else "No"
        ])
    
    writer.writerow([])
    
    # Similarity Analysis
    if similarity_data:
        writer.writerow(['SIMILARITY ANALYSIS'])
        writer.writerow(['File 1', 'File 2', 'Similarity Score'])
        for sim in similarity_data:
            writer.writerow([sim['file1'], sim['file2'], sim['similarity']])
    
    writer.writerow([])
    
    # Summary
    writer.writerow(['SUMMARY'])
    total_files = len(results)
    avg_ai_score = sum(ai_results.get(r['filename'], {}).get('ai_score', 0) for r in results) / total_files if total_files > 0 else 0
    published_count = sum(1 for r in results if ai_results.get(r['filename'], {}).get('published_indicators', False))
    writer.writerow(['Total Files Analyzed', total_files])
    writer.writerow(['Average AI Score', f"{avg_ai_score:.2f}%"])
    writer.writerow(['Papers with Published Indicators', published_count])
    
    return output.getvalue()

def generate_text_report(results, similarity_data, ai_results):
    """Generate text report"""
    report = []
    report.append("PDF ANALYSIS REPORT")
    report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 50)
    report.append("")
    
    report.append("INDIVIDUAL FILE ANALYSIS:")
    report.append("-" * 30)
    for result in results:
        filename = result['filename']
        ai_info = ai_results.get(filename, {})
        report.append(f"File: {filename}")
        report.append(f"  Format: {result['format']} (Confidence: {result['confidence']})")
        report.append(f"  AI Detection: {ai_info.get('ai_score', 0)}% AI, {ai_info.get('human_score', 100)}% Human")
        report.append(f"  Confidence: {ai_info.get('confidence', 'Unknown')}")
        report.append(f"  Published Indicators: {'Yes' if ai_info.get('published_indicators', False) else 'No'}")
        report.append("")
    
    if similarity_data:
        report.append("SIMILARITY ANALYSIS:")
        report.append("-" * 30)
        for sim in similarity_data:
            report.append(f"{sim['file1']} vs {sim['file2']}: {sim['similarity']} similarity")
        report.append("")
    
    report.append("SUMMARY:")
    report.append("-" * 30)
    total_files = len(results)
    avg_ai_score = sum(ai_results.get(r['filename'], {}).get('ai_score', 0) for r in results) / total_files if total_files > 0 else 0
    published_count = sum(1 for r in results if ai_results.get(r['filename'], {}).get('published_indicators', False))
    report.append(f"Total Files Analyzed: {total_files}")
    report.append(f"Average AI Score: {avg_ai_score:.2f}%")
    report.append(f"Papers with Published Indicators: {published_count}")
    
    return "\n".join(report)

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded_files = request.files.getlist('pdfs')
    results, texts, filenames = [], [], []
    ai_results = {}

    for file in uploaded_files:
        if file and file.filename.lower().endswith('.pdf'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            text = extract_text_from_pdf(path)
            paper_format, confidence = detect_format(text)
            
            # Enhanced AI Detection
            ai_analysis = detect_ai_content(text)
            ai_results[file.filename] = ai_analysis
            
            results.append({
                'filename': file.filename,
                'format': paper_format,
                'confidence': f"{confidence}%",
                'ai_score': f"{ai_analysis['ai_score']}%",
                'human_score': f"{ai_analysis['human_score']}%",
                'ai_confidence': ai_analysis['confidence'],
                'published_indicators': ai_analysis.get('published_indicators', False),
                'can_submit': ai_analysis['ai_score'] <= 25  # More strict threshold for submission
            })
            texts.append(text)
            filenames.append(file.filename)

    # Compute similarity
    sim_data = []
    if len(texts) > 1:
        sim_matrix = compute_similarity(texts)
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                sim_data.append({
                    'file1': filenames[i],
                    'file2': filenames[j],
                    'similarity': f"{sim_matrix[i][j]*100:.2f}%"
                })

    # Save submission
    submission_id = save_submission_data(filenames, results, sim_data, ai_results)

    return jsonify({
        'results': results, 
        'similarity': sim_data,
        'submission_id': submission_id,
        'ai_analysis': ai_results
    })

@app.route('/submit-paper', methods=['POST'])
def submit_paper_route():
    """Submit individual paper for approval"""
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    # Find the file and its analysis results
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # In a real application, you'd retrieve the analysis results from your database
    # For now, we'll re-analyze the file
    text = extract_text_from_pdf(file_path)
    paper_format, confidence = detect_format(text)
    ai_analysis = detect_ai_content(text)
    
    # Submit the paper
    submission_result = submit_paper(filename, file_path, {
        'format': paper_format,
        'confidence': confidence
    }, ai_analysis)
    
    return jsonify({
        'message': 'Paper submitted successfully',
        'submission': submission_result,
        'confidence_popup': {
            'ai_score': ai_analysis['ai_score'],
            'human_score': ai_analysis['human_score'],
            'confidence': ai_analysis['confidence'],
            'published_indicators': ai_analysis.get('published_indicators', False),
            'status': submission_result['status']
        }
    })

@app.route('/download-report', methods=['POST'])
def download_report():
    data = request.json
    report_type = data.get('type', 'csv')
    results = data.get('results', [])
    similarity = data.get('similarity', [])
    ai_analysis = data.get('ai_analysis', {})
    
    if report_type == 'csv':
        csv_data = generate_csv_report(results, similarity, ai_analysis)
        output = BytesIO()
        output.write(csv_data.encode('utf-8'))
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name=f"pdf_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mimetype='text/csv'
        )
    else:
        text_data = generate_text_report(results, similarity, ai_analysis)
        output = BytesIO()
        output.write(text_data.encode('utf-8'))
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name=f"pdf_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mimetype='text/plain'
        )

# ---------------- APP ENTRY POINT ----------------
if __name__ == '__main__':
    # If Tesseract is not in PATH, set it here
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)