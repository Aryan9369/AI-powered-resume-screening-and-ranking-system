# AI-Powered Resume Screening & Ranking System
A Web Application for Automated Resume Evaluation

![Project Screenshot](https://github.com/Aryan9369/AI-powered-resume-screening-and-ranking-system/blob/1b54653615026be43fa7ab7d512badd6dcbfac35/Screenshot%20(709).png)

---

## 🚀 Overview

The AI-Powered Resume Screening & Ranking System is a web application designed to automate the initial resume screening process. It analyzes resumes against a given job description and ranks candidates based on relevance, helping reduce manual effort in shortlisting.

This project focuses on backend development, data processing, and foundational NLP techniques, making it suitable for learning and real-world HR screening use cases.

![Dashboard Screenshot](https://github.com/Aryan9369/AI-powered-resume-screening-and-ranking-system/blob/1b54653615026be43fa7ab7d512badd6dcbfac35/Screenshot%20(712).png)

---

## 🎯 Key Features

- 📄 Resume parsing to extract skills, education, and experience  
- 🏆 Automated resume ranking based on job description relevance  
- 🔍 Keyword and text similarity matching using NLP  
- 📊 Interactive dashboard for reviewing ranked resumes  
- 📁 Supports PDF and DOCX resume formats  

---

## 🛠️ Tech Stack

**Frontend**
- Streamlit  

**Backend**
- Python  
- Flask / Django (MVC-based structure)  

**AI & NLP**
- Scikit-learn  
- TF-IDF based feature extraction  

**Data Processing**
- Pandas  
- NLTK / SpaCy  

**Database (optional / extensible)**
- MySQL / PostgreSQL / MongoDB  

**Deployment (optional)**
- AWS / GCP / Heroku  

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Aryan9369/AI-powered-resume-screening-and-ranking-system.git

# Navigate to the project directory
cd AI-powered-resume-screening-and-ranking-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py


🚀 How It Works

Upload resumes in PDF or DOCX format

Enter job description and required skills

The system preprocesses and analyzes resumes

Resumes are scored and ranked automatically

Results are displayed through an interactive dashboard
