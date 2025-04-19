from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle

app = Flask(__name__)

# Load models===========================================================================================================
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))


# Clean resume==========================================================================================================
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# Prediction and Category Name
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category


def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job


def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text


# Resume parsing functions
def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return contact_number


def extract_email_from_resume(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()
    return email


def extract_skills_from_resume(text):
    skills_list = ['Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management',
                   'Deep Learning', 'SQL',
                   'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Node.js', 'MongoDB', 'Express.js', 'Git',
                   'Research',
                   'Data Visualization', 'Matplotlib', 'TensorFlow', 'PyTorch', 'Selenium', 'JUnit',
                   'Software Development', 'Web Development']
    skills = [skill for skill in skills_list if re.search(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE)]
    return skills


def extract_education_from_resume(text):
    education_keywords = ['Computer Science', 'Information Technology', 'Software Engineering',
                          'Mechanical Engineering', 'Electrical Engineering', 'Data Science', 'Cybersecurity',
                          'Business Administration']
    education = [keyword for keyword in education_keywords if re.search(r"(?i)\b" + re.escape(keyword) + r"\b", text)]
    return education


def extract_name_from_resume(text):
    name = None
    pattern = r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b"
    match = re.search(pattern, text)
    if match:
        name = match.group()
    return name


# Routes===================================================================

@app.route('/')
def resume():
    return render_template("resume.html")


@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

        # Predictions and extractions
        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text)

        return render_template('resume.html', predicted_category=predicted_category, recommended_job=recommended_job,
                               phone=phone, name=name, email=email, extracted_skills=extracted_skills,
                               extracted_education=extracted_education)
    else:
        return render_template("resume.html", message="No resume file uploaded.")


if __name__ == '__main__':
    app.run(debug=True)
