# app.py
from flask import Flask, render_template, request, jsonify, session, url_for, redirect
from dotenv import load_dotenv
import os
import json
import uuid
import requests
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
import io
import time
import threading

load_dotenv()

app = Flask(__name__)
app.secret_key = "hr_agent_secret_key_change_this"  # Change this in production

# Configure upload folder
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store job descriptions and CVs in memory (for simplicity)
# In production, consider using a database
job_descriptions = {}
cv_storage = {}
results_storage = {}

# Load prompt template - This will be sent to the LLM
PROMPT_TEMPLATE = """
You are an HR agent named CVision that evaluates CVs against job descriptions. Your task is to analyze the following CV and job description, then provide a detailed assessment.

### JOB DESCRIPTION:
{{JOB_DESCRIPTION}}

### CV CONTENT:
{{CV_CONTENT}}

### INSTRUCTIONS:
Evaluate how well this candidate matches the job requirements. Provide a thorough analysis and scoring.

Your response must be in valid JSON format following this exact structure:
{
  "candidate_name": "Extract the candidate's full name from the CV",
  "overall_score": 0, // Score from 1-10
  "skills_match_score": 0, // Score from 1-10
  "experience_relevance_score": 0, // Score from 1-10
  "education_qualifications_score": 0, // Score from 1-10
  "potential_cultural_fit_score": 0, // Score from 1-10
  "strengths": [
    "List 3-5 strengths relevant to the job"
  ],
  "weaknesses": [
    "List 2-3 weaknesses or improvement areas"
  ],
  "recommendation": "Choose one: Shortlist / Consider / Reject",
  "reasoning": "Provide a 2-3 sentence explanation for your recommendation"
}

Important: Your entire response must be a valid JSON object that can be parsed. Do not include any text outside the JSON structure.
"""

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    """Extract text from a DOCX file."""
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file):
    """Extract text from uploaded file based on extension."""
    try:
        file_content = file.read()
        file.seek(0)  # Reset file pointer for potential future reads
        
        filename = secure_filename(file.filename)
        print(f"Extracting text from file: {filename}")
        
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(io.BytesIO(file_content))
            print(f"Extracted {len(text)} characters from PDF")
            print(f"File exctracted successfully")
            print(text)
            return text
        elif filename.lower().endswith('.docx'):
            text = extract_text_from_docx(io.BytesIO(file_content))
            print(f"Extracted {len(text)} characters from DOCX")
            print(text)
            return text
        elif filename.lower().endswith('.txt'):
            text = file_content.decode('utf-8')
            print(f"Extracted {len(text)} characters from TXT")
            print(text)
            return text
        else:
            print(f"Unsupported file format: {filename}")
            return "Unsupported file format"
    except Exception as e:
        print(f"Error extracting text from {file.filename}: {str(e)}")
        raise

def evaluate_cv_with_ollama(cv_content, job_description):
    """Evaluate a CV using locally running Ollama."""
    # Replace placeholders in prompt template
    prompt = PROMPT_TEMPLATE.replace("{{JOB_DESCRIPTION}}", job_description)
    prompt = prompt.replace("{{CV_CONTENT}}", cv_content)
    
    try:
        # Call Ollama API (assuming it's running locally)
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "llama3",  # Or another model you have pulled
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result_text = response.json().get("response", "")
            
            # Extract JSON from response
            try:
                # Try to find JSON object in the text
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result_json = json.loads(json_match.group())
                    return result_json
                else:
                    return {"error": "Could not find JSON in response", "raw_response": result_text}
            except json.JSONDecodeError:
                return {"error": "Could not parse JSON response", "raw_response": result_text}
        else:
            return {"error": f"API Error: {response.status_code}", "raw_response": response.text}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

def evaluate_cv_with_openai(cv_content, job_description):
    """Evaluate a CV using OpenAI API."""
    print("Evaluating CV with OpenAI API...")
    
    try:
        # Validate inputs
        if not cv_content or len(cv_content.strip()) < 10:
            print("Error: CV content is empty or too short")
            return {"error": "CV content is invalid"}
            
        if not job_description or len(job_description.strip()) < 10:
            print("Error: Job description is empty or too short")
            return {"error": "Job description is invalid"}
            
        # Replace placeholders in prompt template
        prompt = PROMPT_TEMPLATE.replace("{{JOB_DESCRIPTION}}", job_description)
        prompt = prompt.replace("{{CV_CONTENT}}", cv_content)
        
        import openai
        import os

        api_key = os.environ.get("GROQ_API_KEY")
        print(f"API Key present: {bool(api_key)}")
        print(f"API Key length: {len(api_key) if api_key else 0}")
        
        if not api_key:
            print("Error: GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY not configured")

        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        
        # Debug logs for request data
        print(f"CV Content length: {len(cv_content)} characters")
        print(f"Job Description length: {len(job_description)} characters")
        print("Sending request to Groq API...")
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert HR assistant that evaluates CVs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        result_text = response.choices[0].message.content
        print(f"API Response received. Length: {len(result_text)} characters")
        print(f"First 200 characters of response: {result_text[:200]}")
        
        try:
            # Try to find JSON object in the text
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                print("Successfully parsed JSON response")
                return result_json
            else:
                print("Could not find JSON in response")
                print(f"Full response: {result_text}")
                return {"error": "Could not find JSON in response", "raw_response": result_text}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            print(f"Problematic text: {result_text}")
            return {"error": "Could not parse JSON response", "raw_response": result_text}
    except Exception as e:
        print(f"Exception in evaluate_cv_with_openai: {str(e)}")
        return {"error": f"Exception: {str(e)}"}

def evaluate_cv_with_anthropic(cv_content, job_description):
    """Evaluate a CV using Claude API from Anthropic."""
    # Replace placeholders in prompt template
    prompt = PROMPT_TEMPLATE.replace("{{JOB_DESCRIPTION}}", job_description)
    prompt = prompt.replace("{{CV_CONTENT}}", cv_content)
    
    try:
        import anthropic
        # Make sure to set your API key: client = anthropic.Anthropic(api_key="your-api-key")
        client = anthropic.Anthropic()
        
        response = client.messages.create(
            model="claude-3-opus-20240229",  # or another appropriate model
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # Lower temperature for more consistent outputs
        )
        
        result_text = response.content[0].text
        
        try:
            # Try to find JSON object in the text
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                return result_json
            else:
                return {"error": "Could not find JSON in response", "raw_response": result_text}
        except json.JSONDecodeError:
            return {"error": "Could not parse JSON response", "raw_response": result_text}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

def evaluate_cv(cv_content, job_description):
    """Choose which LLM to use for evaluation based on configuration."""
    # You can change this to select which LLM you want to use
    llm_choice = "openai"  # Options: "ollama", "openai", "anthropic"
    
    if llm_choice == "ollama":
        return evaluate_cv_with_ollama(cv_content, job_description)
    elif llm_choice == "openai":
        return evaluate_cv_with_openai(cv_content, job_description)
    elif llm_choice == "anthropic":
        return evaluate_cv_with_anthropic(cv_content, job_description)
    else:
        return {"error": f"Unknown LLM choice: {llm_choice}"}

def process_cv_batch(job_id, cv_ids):
    """Process a batch of CVs against a job description."""
    print(f"Processing CV batch for job {job_id}")
    try:
        job_description = job_descriptions.get(job_id)
        if not job_description:
            print(f"No job description found for job {job_id}")
            results_storage[job_id] = [{"error": "No job description found"}]
            return
        
        results = []
        
        for cv_id in cv_ids:
            print(f"Processing CV {cv_id}")
            try:
                cv_data = cv_storage.get(cv_id, {})
                cv_content = cv_data.get("content")
                cv_filename = cv_data.get("filename")
                
                if not cv_content:
                    print(f"No content found for CV {cv_id}")
                    results.append({
                        "error": "No CV content found",
                        "cv_id": cv_id,
                        "cv_filename": cv_filename,
                        "overall_score": 0
                    })
                    continue
                
                # Evaluate the CV using the chosen LLM
                result = evaluate_cv(cv_content, job_description)
                print(f"Evaluation result for CV {cv_id}: {result.get('error', 'Success')}")
                
                if "error" in result:
                    print(f"Error processing CV {cv_id}: {result['error']}")
                    result["cv_id"] = cv_id
                    result["cv_filename"] = cv_filename
                    result["overall_score"] = 0
                else:
                    result["cv_id"] = cv_id
                    result["cv_filename"] = cv_filename
                results.append(result)
                
            except Exception as e:
                print(f"Exception processing CV {cv_id}: {str(e)}")
                results.append({
                    "error": f"Exception: {str(e)}",
                    "cv_id": cv_id,
                    "cv_filename": cv_filename,
                    "overall_score": 0
                })
        
        # Sort results by overall score (descending)
        results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        # Store results
        print(f"Storing {len(results)} results for job {job_id}")
        results_storage[job_id] = results
        
    except Exception as e:
        print(f"Exception in process_cv_batch: {str(e)}")
        results_storage[job_id] = [{"error": f"Processing failed: {str(e)}"}]

@app.route('/')
def index():
    return app.send_file('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload_cvs():
    """Page for uploading CVs."""
    job_id = session.get('current_job_id')
    job_title = session.get('current_job_title')
    
    if not job_id or not job_title:
        return redirect(url_for('index'))
    
    return render_template('upload.html', job_title=job_title)

@app.route('/create-job', methods=['POST'])
def create_job():
    """Create a new job posting and store the job description."""
    job_title = request.form.get('job_title')
    job_description = request.form.get('job_description')
    
    print(f"Creating job: {job_title}")
    print(f"Job description length: {len(job_description) if job_description else 0}")
    
    if not job_title or not job_description:
        return jsonify({"error": "Job title and description are required"}), 400
    
    # Generate a unique ID for the job
    job_id = str(uuid.uuid4())
    
    # Store the job description
    job_descriptions[job_id] = job_description
    print(f"Stored job description with ID: {job_id}")
    
    # Store job ID in session for use in subsequent requests
    session['current_job_id'] = job_id
    session['current_job_title'] = job_title
    
    return redirect(url_for('upload_cvs'))

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    """Handle CV file upload."""
    job_id = session.get('current_job_id')
    print(f"Handling CV upload for job {job_id}")
    
    if not job_id:
        print("Error: No active job")
        return jsonify({"error": "No active job"}), 400
    
    if 'cv_files' not in request.files:
        print("Error: No files provided")
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('cv_files')
    uploaded_files = []
    cv_ids = []  # Track CV IDs for this upload
    
    for file in files:
        if file.filename == '':
            continue
        
        try:
            print(f"Processing file: {file.filename}")
            # Extract text content from the file
            cv_content = extract_text_from_file(file)
            print(f"Extracted content length: {len(cv_content) if cv_content else 0}")
            
            # Generate ID for the CV
            cv_id = str(uuid.uuid4())
            print(f"Generated CV ID: {cv_id}")
            
            # Store CV content in memory
            cv_storage[cv_id] = {
                "filename": file.filename,
                "content": cv_content,
                "job_id": job_id
            }
            
            cv_ids.append(cv_id)  # Add to list of IDs for this upload
            uploaded_files.append({
                "id": cv_id,
                "filename": file.filename
            })
            
        except Exception as e:
            print(f"Error processing file {file.filename}: {str(e)}")
            return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 500
    
    # Update session with new CV IDs
    session['cv_ids'] = cv_ids  # Replace existing IDs with new ones
    print(f"Updated session with {len(cv_ids)} CV IDs: {cv_ids}")
    
    return jsonify({"success": True, "files": uploaded_files})

@app.route('/process', methods=['POST'])
def process_cvs():
    """Process uploaded CVs against the job description."""
    job_id = session.get('current_job_id')
    cv_ids = session.get('cv_ids', [])
    
    if not job_id:
        return jsonify({"error": "No active job"}), 400
    
    if not cv_ids:
        return jsonify({"error": "No CVs uploaded"}), 400
    
    # Start processing in a background thread
    thread = threading.Thread(target=process_cv_batch, args=(job_id, cv_ids))
    thread.start()
    
    return jsonify({"success": True, "job_id": job_id})

@app.route('/results')
def show_results():
    """Show results page."""
    job_id = session.get('current_job_id')
    job_title = session.get('current_job_title')
    
    if not job_id:
        return redirect(url_for('index'))
    
    return render_template('results.html', job_title=job_title)

@app.route('/api/results')
def get_results():
    """API endpoint to get processing results."""
    job_id = session.get('current_job_id')
    
    if not job_id:
        return jsonify({"error": "No active job"}), 400
    
    results = results_storage.get(job_id)
    
    if results is None:
        return jsonify({"status": "processing"})
    
    return jsonify({"status": "completed", "results": results})

@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset the current session."""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)