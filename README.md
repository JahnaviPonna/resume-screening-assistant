# 📄 Resume Screening Assistant

An AI-powered web application that automatically analyzes uploaded resumes and predicts the most suitable job role using semantic similarity, keyword matching, and job title detection.

The application supports PDF and Word documents, including scanned PDFs through OCR, and provides confidence scores, matched keywords, top role predictions, and downloadable CSV reports.

---

## 🚀 Features

- Upload one or multiple resumes
- Supports **PDF**, **DOCX**, and **DOC** files
- OCR support for scanned/image-based PDFs
- Predicts among **20 different job roles**
- Displays:
  - Predicted job role
  - Confidence score
  - Matched keywords
  - Top 3 closest role matches
- Interactive bar chart showing role distribution
- Export classification results as CSV
- Marks low-confidence resumes as **Unclassified**

---

## 🧠 Classification Method

The prediction combines three different signals:

| Method | Weight |
|----------|---------|
| Semantic Similarity (Sentence Transformers) | 60% |
| Keyword Matching | 30% |
| Job Title Alias Detection | 10% |

The application uses the **all-mpnet-base-v2** Sentence Transformer model to compare resume content with predefined job role descriptions.

---

# 📂 Project Structure

```
Resume_screening_assistant/
│
├── app.py
├── requirements.txt
├── uploads/
├── templates/
│   └── index.html
└── README.md
```

> **Note:** `index.html` must be inside the `templates` folder.

---

# 💻 Requirements

- Python 3.9+
- Windows/Linux/macOS
- pip
- Tesseract OCR (for scanned PDFs)
- Internet connection (first run only)

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd Resume_screening_assistant
```

---

## 2. Create a Virtual Environment (Recommended)

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Tesseract OCR

Download and install:

https://github.com/UB-Mannheim/tesseract/wiki

After installation, verify:

```bash
tesseract --version
```

---

## 4. Install PyTorch (CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Install Project Dependencies

Using requirements file:

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install flask sentence-transformers pandas pdfplumber PyMuPDF pytesseract Pillow python-docx fuzzywuzzy python-Levenshtein
```

---

## 6. Run the Application

```bash
python app.py
```

The first launch downloads the Sentence Transformer model (~420 MB).

Open your browser:

```
http://127.0.0.1:5000
```

---

# 📁 Supported Resume Formats

| Format | Support |
|----------|----------|
| PDF | ✅ Text PDFs |
| Scanned PDF | ✅ OCR |
| DOCX | ✅ |
| DOC | ✅ |

---

# 🎯 Supported Job Roles

The classifier predicts the following roles:

- Software Developer
- Web Developer
- Data Scientist
- AI/ML Engineer
- DevOps Engineer
- Cloud Engineer
- Data Analyst
- Database Administrator
- Software Tester / QA
- UI/UX Designer
- Cybersecurity Analyst
- Network Engineer
- Embedded Engineer
- VLSI Design Engineer
- Mechanical Engineer
- Electrical Engineer
- Business Analyst
- Research Scientist
- Technical Writer
- Product Manager

If a resume does not match any role with sufficient confidence, it is classified as **Unclassified**.

---

# 📊 Output

For every uploaded resume, the application provides:

- Predicted Job Role
- Confidence Score
- Matched Keywords
- Top 3 Similar Roles
- CSV Export
- Role Distribution Chart

---

# 🛠 Technologies Used

- Python
- Flask
- Sentence Transformers (`all-mpnet-base-v2`)
- PyTorch
- pdfplumber
- PyMuPDF
- pytesseract
- Pillow
- python-docx
- pandas
- fuzzywuzzy
- Bootstrap 5
- Chart.js

---

# 📦 Project Files

| File | Description |
|------|-------------|
| `app.py` | Flask backend and classification logic |
| `templates/index.html` | Frontend user interface |
| `requirements.txt` | Project dependencies |
| `uploads/` | Stores uploaded resumes temporarily |
| `resume_classification.csv` | Generated classification results |

---

# ❗ Troubleshooting

### Tesseract not recognized

Ensure Tesseract is installed and added to your system PATH.

Verify:

```bash
tesseract --version
```

---

### Missing Python Package

Install dependencies again:

```bash
pip install -r requirements.txt
```

---

### TemplateNotFound Error

Ensure the folder structure is:

```
templates/
    index.html
```

---

### Port Already in Use

Change the port inside `app.py`:

```python
app.run(debug=True, port=8000)
```

Then open:

```
http://127.0.0.1:8000
```

---

### First Run is Slow

This is expected. The AI model is downloaded and cached during the first execution.

---

# 👩‍💻 Author

**Jahnavi Sushma Priya Ponna**

📧 Email: **jahnavispponna@gmail.com**
