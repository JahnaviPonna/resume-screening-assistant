from flask import Flask, render_template, request, send_file, jsonify
import os
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

job_roles = {
    "Software Developer": "Java, C++, Python, software engineering, APIs, backend, frontend, frameworks",
    "Data Scientist": "machine learning, pandas, statistics, Python, data analysis, visualization, models",
    "Web Developer": "HTML, CSS, JavaScript, React, Node.js, web design, responsive layout",
    "AI/ML Engineer": "deep learning, PyTorch, TensorFlow, model training, AI projects",
    "UI/UX Designer": "Figma, Adobe XD, wireframes, user experience, prototyping, visual design",
    "Cybersecurity Analyst": "penetration testing, firewalls, encryption, ethical hacking, risk analysis",
    "Embedded Engineer": "microcontrollers, Arduino, embedded C, sensors, real-time systems",
    "VLSI Design Engineer": "Verilog, VHDL, ASIC, FPGA, synthesis, layout, chip design",
    "Mechanical Engineer": "CAD, SolidWorks, thermodynamics, mechanical design, manufacturing",
    "Electrical Engineer": "circuit analysis, MATLAB, power systems, signal processing",
    "Electronic Engineer": "digital circuits, semiconductors, embedded systems, PCB design",
    "Software Tester": "manual testing, automation, Selenium, JUnit, bug reports",
    "Chip Designer": "logic synthesis, physical design, layout, RTL coding, DFT",
    "Ethical Hacker": "penetration testing, cybersecurity tools, network scanning, vulnerability testing",
    "Network Engineer": "routers, switches, subnetting, network setup, protocols",
    "Database Administrator": "MySQL, PostgreSQL, database design, indexing, queries, backups",
    "Systems Analyst": "requirements analysis, process modeling, system specs, IT solutions",
    "Technical Consultant": "IT advisory, solution design, client engagement, enterprise systems",
    "Business Analyst": "business requirements, stakeholder analysis, process optimization, documentation",
    "Quality Assurance Engineer": "QA testing, test plans, automation, validation",
    "DevOps Engineer": "CI/CD, Jenkins, Docker, Kubernetes, version control, automation",
    "Cloud Engineer": "AWS, Azure, GCP, cloud architecture, deployment",
    "Data Analyst": "data cleaning, Excel, SQL, visualization, business insights",
    "Research Scientist": "literature review, experiments, scientific method, publication, analysis",
    "Technical Writer": "documentation, user manuals, API docs, clear writing",
    "Sales Engineer": "client demos, product knowledge, tech specs, sales cycle",
    "Field Engineer": "on-site setup, troubleshooting, technical support",
    "Operations Manager": "logistics, workflows, efficiency, process improvement",
    "Supply Chain Manager": "inventory, procurement, distribution, demand planning",
    "Consultant": "strategic advisory, domain expertise, transformation, analysis",
    "Entrepreneur": "startup, product development, pitching, growth, leadership",
    "Professor/Educator": "teaching, curriculum, mentoring, academic projects"
}

role_embeddings = {role: model.encode(desc, convert_to_tensor=True) for role, desc in job_roles.items()}


def extract_text(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + ' '
    return text.strip()


def classify_resume(text):
    text = text[:6000].lower()
    resume_embedding = model.encode(text, convert_to_tensor=True)
    similarities, match_keywords = {}, {}

    for role, desc in job_roles.items():
        embedding = role_embeddings[role]
        sim_score = util.cos_sim(resume_embedding, embedding).item()
        similarities[role] = sim_score
        keywords = [kw.strip().lower() for kw in desc.split(',')]
        matched = [kw for kw in keywords if fuzz.partial_ratio(kw, text) > 85]
        match_keywords[role] = matched

    best_role = max(similarities, key=similarities.get)
    rationale = ', '.join(match_keywords[best_role]) if match_keywords[best_role] else 'No keywords matched'
    return best_role, rationale


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('resumes')
    results = []

    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        text = extract_text(path)
        role, rationale = classify_resume(text)
        results.append({'filename': file.filename, 'category': role, 'reason': rationale})

    df = pd.DataFrame(results)
    df.to_csv('resume_classification.csv', index=False)
    return jsonify(results)


@app.route('/visualize')
def visualize():
    df = pd.read_csv('resume_classification.csv')
    data = df['category'].value_counts().to_dict()
    return jsonify(data)


@app.route('/download')
def download():
    return send_file('resume_classification.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



# pip install flask sentence-transformers pandas matplotlib PyPDF2 fuzzywuzzy python-Levenshtein
# python app.py
