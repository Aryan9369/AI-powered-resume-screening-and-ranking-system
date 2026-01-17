# AI-Powered Resume Screening & Ranking System
A Web Application for Automated Resume Evaluation

![Project Screenshot](https://github.com/Aryan9369/AI-powered-resume-screening-and-ranking-system/blob/1b54653615026be43fa7ab7d512badd6dcbfac35/Screenshot%20(709).png)

## 🚀 Overview
The AI-Powered Resume Screening & Ranking System is a web application designed to automate the initial resume screening process. It reduces manual effort by analyzing resumes against a given job description and ranking candidates based on relevance.

This project mainly focuses on backend logic, data processing, and foundational NLP techniques, making it suitable for real-world HR screening workflows and learning-oriented backend development.

![Dashboard Screenshot](https://github.com/Aryan9369/AI-powered-resume-screening-and-ranking-system/blob/1b54653615026be43fa7ab7d512badd6dcbfac35/Screenshot%20(712).png)

## 🎯 Key Features
- 📄 Resume parsing to extract skills, education, and experience
- 🏆 Automated resume ranking based on job description relevance
- 🔍 Keyword and text similarity matching using NLP
- 📊 Interactive dashboard for uploading and reviewing resumes
- 📁 Supports PDF and DOCX resume formats

## 🛠️ Tech Stack
**Frontend**
- Streamlit

**Backend**
- Python
- Flask / Django (MVC-based project structure)

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



## 🚀 How It Works
1. Upload resumes in PDF or DOCX format  
2. Enter job description and required skills  
3. The system preprocesses and analyzes resumes  
4. Resumes are scored and ranked automatically  
5. Results are displayed through an interactive dashboard  

## 📁 Project Structure



AI-powered-resume-screening/
│── models/ # ML and NLP scripts
│── data/ # Sample resumes / datasets
│── templates/ # Backend templates (Flask/Django)
│── static/ # Static assets (if applicable)
│── app.py # Main Streamlit application
│── requirements.txt
│── README.md


## 🤖 AI / NLP Approach
- Text preprocessing and cleaning  
- Feature extraction using TF-IDF  
- Similarity scoring between resumes and job description  
- Ranking based on relevance scores  

> Note: This project focuses on foundational NLP and backend integration rather than advanced deep-learning models.

## 🎯 Future Enhancements
- Integration of transformer-based models (BERT / LLMs)  
- Improved semantic skill matching  
- Cloud deployment with scalable backend APIs  

## 📜 License
This project is licensed under the MIT License.

## 👤 Author
**Aryan Katiyar**  
📧 Email: aryankatiyarjnv@gmail.com  
💼 LinkedIn: https://www.linkedin.com/in/aryan-katiyar-225893214  
💻 GitHub: https://github.com/Aryan9369
