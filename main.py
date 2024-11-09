# main.py

from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pdf2image
import requests
import base64
import openai
import re
import uuid
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
import json
import logging
import concurrent.futures

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
DATABASE_URI = 'sqlite:///quizify.db'

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS", "DELETE", "PUT"])

db = SQLAlchemy(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Firebase Admin SDK
SERVICE_ACCOUNT_KEY = 'service_key.json'  # Ensure this path is correct

try:
    if not os.path.exists(SERVICE_ACCOUNT_KEY):
        logger.error(f"Service account key file not found at {SERVICE_ACCOUNT_KEY}")
        exit(1)
    else:
        logger.info(f"Service account key file found at {SERVICE_ACCOUNT_KEY}")

    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
    exit(1)

# Initialize OpenAI Client
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.error("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

client = openai

# Database Models
class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(150), nullable=False)  # Firebase UID
    filename = db.Column(db.String(150), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    quiz = db.Column(db.Text, nullable=False)  # Store quiz as JSON string

# Firebase Authentication Decorator
def firebase_token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            id_token = auth_header.split('Bearer ')[1]
            try:
                decoded_token = auth.verify_id_token(id_token)
                request.user = decoded_token  # Attach user info to the request
            except Exception as e:
                logger.error(f"Token verification failed: {e}")
                return jsonify({'error': 'Invalid token'}), 401
        else:
            logger.error("Authorization header missing or malformed.")
            return jsonify({'error': 'Authorization header missing or malformed'}), 401
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Quizify Backend API!", 200

# MathPix OCR function to handle math-heavy content
def mathpix_ocr(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode()

    url = "https://api.mathpix.com/v3/text"
    headers = {
        "app_id": os.getenv('MATHPIX_APP_ID'),  # Set your MathPix app ID as an environment variable
        "app_key": os.getenv('MATHPIX_APP_KEY'),  # Set your MathPix app key as an environment variable
        "Content-type": "application/json"
    }
    data = {
        "src": f"data:image/jpeg;base64,{image_base64}",
        "formats": ["text", "latex_styled"],
        "data_options": {
            "include_asciimath": True
        }
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get("text", "No text extracted")
        else:
            logger.error(f"MathPix Error: {response.status_code} - {response.text}")
            return f"Error: {response.json()}"
    except requests.exceptions.RequestException as e:
        logger.error(f"MathPix Request Exception: {e}")
        return f"Error: {e}"

def use_gpt_to_extract_questions(text):
    prompt = f"""
    You are an AI trained to extract quiz questions from academic texts. Below is an academic text, and your task is to identify and extract questions in a structured manner.

    Here is how you should structure the output:

    1. Each **main question** should be numbered (e.g., "1.", "2.") and clearly stated, followed by a list of **sub-questions** if there are any.
    2. **Sub-questions** should be grouped under their main question and listed as "(a)", "(b)", "(c)" format under the relevant main question.
    3. If there are multiple parts to the same question (e.g., 4a, 4b, 4c), they should be output as follows:

       - Main question number: followed by the question
       - Sub-questions should be indented under it.

    Example Output:
    ```
    1. Describe the properties of water.
       (a) What is the boiling point of water?
       (b) Explain why water is a universal solvent.

    2. Translate the following statements into logical expressions:
       (a) All humans are mortal.
       (b) Socrates is a human.
    ```

    Here is the academic text:

    {text}

    Extract and format the questions exactly as described above.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts questions from text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5,
        )

        # Get the response content and remove unwanted backticks
        raw_questions = response.choices[0].message.content.strip().replace('```', '')

        # Split the response by newlines that signify new questions
        questions = re.split(r'\n(?=\d+\.)', raw_questions)
        questions = [q.strip() for q in questions if q.strip()]

        return questions
    except Exception as e:
        logger.error(f"Error with OpenAI API: {str(e)}")
        return f"Error with OpenAI API: {str(e)}"

def generate_hints_for_questions(questions):
    prompt = "Provide a useful hint for each of the following questions that helps guide towards the answer without giving it away:\n\n"
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\nHint:"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that provides helpful hints for quiz questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150 * len(questions),  # Allocate enough tokens for all hints
            temperature=0.7,
        )

        hints_text = response.choices[0].message.content.strip()
        # Split the response by newlines that signify individual hints
        hints = re.split(r'\n(?=\d+\.)', hints_text)
        hints = [re.sub(r'^Hint[:\-]?\s*', '', hint, flags=re.IGNORECASE).strip() for hint in hints]

        # Ensure the number of hints matches the number of questions
        if len(hints) != len(questions):
            logger.warning("Number of hints does not match number of questions.")
            # Fill the remaining hints with a default message
            while len(hints) < len(questions):
                hints.append("No hint available.")

        return hints

    except Exception as e:
        logger.error(f"Error generating hints: {e}")
        # Return default hints if hint generation fails
        return ["No hint available."] * len(questions)

def process_page(page):
    """Helper function to process a single PDF page with MathPix OCR."""
    try:
        page_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_page_{uuid.uuid4()}.jpg")
        page.save(page_file_path, "JPEG")
        logger.info(f"Processing {page_file_path}")
        page_text = mathpix_ocr(page_file_path)
        os.remove(page_file_path)
        return page_text
    except Exception as e:
        logger.error(f"Error processing page: {e}")
        return ""

@app.route('/upload', methods=['POST'])
@firebase_token_required
def upload_file():
    user_id = request.user['uid']

    if 'file' not in request.files:
        logger.error("No file part in the request.")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file.")
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(file_path)
            logger.info(f"File saved to {file_path}")
        except Exception as e:
            logger.error(f"File save error: {e}")
            return jsonify({'error': f'File save error: {str(e)}'}), 500

        # Process the uploaded file
        try:
            extracted_text = ""
            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension in ['pdf', 'png', 'jpg', 'jpeg']:
                if file_extension == 'pdf':
                    images = pdf2image.convert_from_path(file_path, dpi=300)
                    logger.info(f"Converted PDF to {len(images)} image(s).")
                    # Use ThreadPoolExecutor to process pages in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        page_texts = list(executor.map(process_page, images))
                    extracted_text = "\n".join(page_texts)
                else:
                    extracted_text = mathpix_ocr(file_path)

            if extracted_text.strip() == "":
                logger.warning("No text could be extracted from the file.")
                return jsonify({'extracted_text': 'No text could be extracted'}), 200

            quiz_questions = use_gpt_to_extract_questions(extracted_text)

            # Check if quiz is a list, else return error
            if not isinstance(quiz_questions, list):
                logger.error("Failed to extract quiz questions.")
                return jsonify({'error': 'Failed to extract quiz questions'}), 500

            # Generate hints for all questions in batch
            hints = generate_hints_for_questions(quiz_questions)

            # Combine questions and hints
            quiz_with_hints = []
            for question, hint in zip(quiz_questions, hints):
                quiz_with_hints.append({
                    'question': question,
                    'hint': hint
                })

            # Save submission to database
            quiz_json = json.dumps(quiz_with_hints)
            new_submission = Submission(user_id=user_id, filename=unique_filename, extracted_text=extracted_text, quiz=quiz_json)
            db.session.add(new_submission)
            db.session.commit()
            logger.info(f"Submission saved for user {user_id}.")

            return jsonify({'extracted_text': extracted_text, 'quiz': quiz_with_hints}), 200

        except Exception as e:
            logger.error(f"Failed to process file: {e}")
            return jsonify({'error': f'Failed to process file: {str(e)}'}), 500
    else:
        logger.error("Invalid file type uploaded.")
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/submissions', methods=['GET'])
@firebase_token_required
def get_submissions():
    user_id = request.user['uid']
    submissions = Submission.query.filter_by(user_id=user_id).all()
    submission_data = []
    for s in submissions:
        try:
            quiz = json.loads(s.quiz)
        except json.JSONDecodeError:
            quiz = s.quiz  # In case quiz is not a valid JSON
        submission_data.append({
            'filename': s.filename,
            'extracted_text': s.extracted_text,
            'quiz': quiz
        })
    logger.info(f"Retrieved {len(submission_data)} submissions for user {user_id}.")
    return jsonify({'submissions': submission_data}), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
