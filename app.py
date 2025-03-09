import streamlit as st
import pandas as pd
from sqlalchemy.orm import sessionmaker
from models import engine, Resume, Base
from resume_parser import parser
import os
import spacy
import logging
import joblib
from config import SKILL_MATCH_WEIGHT, EXPERIENCE_WEIGHT, ML_MODEL_WEIGHT, MODEL_PATH
import plotly.graph_objects as go  # For visualizations



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy's English model
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully.")
except OSError as e:
    logging.error(f"Error loading spaCy model: {e}")
    st.error(f"Error: spaCy model not found. Please run: `python -m spacy download en_core_web_sm`")
    nlp = None # Disable NLP if model not loaded

# Database connection setup
Session = sessionmaker(bind=engine)

# Load the trained machine learning model
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Machine learning model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading machine learning model: {e}")
    st.error(f"Error: ML model not found. Please train the model first (python ml_model/train_model.py)")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'docx', 'pdf', 'txt'}

def process_resume(file_content, filename, file_type):
    """Extracts info, stores to database."""
    try:
        with Session() as session:
            if file_type == "docx":
                text = parser.extract_text_from_docx(file_content)
            elif file_type == "pdf":
                text = parser.extract_text_from_pdf(file_content)
            elif file_type == "txt":
                text = file_content.decode("utf-8")
            else:
                st.warning(f"Unsupported file type for {filename}. Skipping.")
                return [], 0, {}, "" #Return a tuple

            if not text:
                st.warning(f"Could not extract text from {filename}. Skipping.")
                return [], 0, {}, "" #Return a tuple

            skills = parser.extract_skills_nlp(text)
            experience = parser.extract_experience_nlp(text)

            # Create a new Resume object and store the data
            resume = Resume(filename=filename, text_content=text, skills=','.join(skills), experience=experience)
            session.add(resume)
            session.commit()

            #Create preview Data
            preview_data = {
              "filename": filename,
              "skills": skills,
              "experience": experience,
            }

            return skills, experience, preview_data, text #Return a tuple

    except Exception as e:
        st.error(f"Error processing resume {filename}: {e}")
        logging.exception(f"Error processing {filename}: {e}")
        return [], 0, {}, "" #Return a tuple


def extract_features(resume, job_keywords):
    """Extracts features for the machine learning model."""
    resume_skills = resume.skills.split(',')
    skill_match_count = sum(1 for skill in resume_skills if skill in job_keywords)
    experience = resume.experience

    # Add more features here (education, job titles, etc.)
    # ...

    return [skill_match_count, experience]  # Must match the features the model was trained on


def calculate_ranking_score(resume, job_keywords, skill_weight, experience_weight, ml_weight):
    """Calculates ranking score based on skill match, experience, and ML prediction."""
    skill_match_count = sum(1 for skill in resume.skills.split(',') if skill in job_keywords)

    # Use Machine Learning Model
    if model:  # Only if the model loaded successfully
        features = extract_features(resume, job_keywords)
        try:
            ml_score = model.predict([features])[0]
            ml_score = min(max(ml_score, 0), 100) #Clamp value
        except Exception as e:
            logging.warning(f"Error during model prediction: {e}")
            ml_score = 0
    else:
        ml_score = 0

    # Final ranking score (weighted combination)
    score = min((skill_match_count * skill_weight) + (resume.experience * experience_weight) + (ml_score * ml_weight), 100)

    return score


def extract_keywords_from_job_description(jd_text):
    """Extracts keywords from the job description using NLP."""
    if not jd_text or not nlp:
        return []  # Return an empty list if the job description is empty or NLP is not loaded

    try:
        doc = nlp(jd_text)
        keywords = set()
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"]:
                keywords.add(token.lemma_)
        return keywords
    except Exception as e:
        st.error(f"Error extracting keywords from job description: {e}")
        logging.exception(f"Error extracting keywords: {e}")
        return []

# Function to create skill matching visualization
def create_skill_matching_chart(resume_skills, job_keywords):
    """Creates a bar chart to visualize skill matching."""
    matched_skills = [skill for skill in resume_skills if skill in job_keywords]
    unmatched_skills = [skill for skill in resume_skills if skill not in job_keywords]

    data = [
        go.Bar(name='Matched Skills', x=matched_skills, y=[1]*len(matched_skills), marker_color='green'),
        go.Bar(name='Unmatched Skills', x=unmatched_skills, y=[1]*len(unmatched_skills), marker_color='red')
    ]

    layout = go.Layout(
        title='Skill Matching Visualization',
        xaxis_title='Skills',
        yaxis_title='Match',
        yaxis=dict(showticklabels=False), # Hide Y-axis labels
        barmode='stack' # Stack the bars
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def main():
    st.title("AI-Powered Resume Screening and Ranking System")

    # Initialize Session State
    if 'uploaded_resumes' not in st.session_state:
        st.session_state['uploaded_resumes'] = []

    # Theme Selection
    theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                color: #ffffff; /* General text color */
                background-color: #262730;
            }
            .stApp {
                background-color: #262730;
            }
            .streamlit-expanderHeader {
                color: #ffffff;
            }
            .stTextInput > label {
                color: #ffffff;
            }
            .stTextArea > label {
                color: #ffffff;
            }
            .stNumberInput > label {
                color: #ffffff;
            }
            .stSlider > label {
                color: #ffffff;
            }
            .stSelectbox > label {
                color: #ffffff;
            }
            .stRadio > label {
                color: #ffffff;
            }
            .css-14xtw13 {
                color: #ffffff; /* Table Header */
            }

            .css-qrbk6 {
                color: #ffffff; /*File text*/
            }

            .css-qrbk6 * {
                color: #ffffff;
            }

            .css-qrbk6 a {
                color: #ffffff; /*Text area link */
            }

            .css-12oz5g7 {
                color: #ffffff; /*The info and error messages */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar for Input and Configuration
    with st.sidebar:
        st.header("Inputs and Configuration")

        # Ranking Weights Explanation
        with st.expander("Ranking Weights Explained"):
            st.write("""
            The ranking score is calculated based on three factors: Skill Match, Experience, and Machine Learning (ML) Model Score.
            Each factor is assigned a weight, which determines its importance in the overall ranking.

            *   **Skill Match Weight:** This weight determines the importance of skills found in the resume that match keywords from the job description. A higher weight means that skills are more important for the ranking.

            *   **Experience Weight:** This weight determines the importance of the years of experience listed on the resume. A higher weight means that experience is more important for the ranking.

            *   **ML Model Weight:** This weight determines the importance of the score predicted by the machine learning model. The ML model is trained on a dataset of resumes and job descriptions, and it learns to predict the suitability of a resume for a given job. A higher weight means that the ML model's prediction is more important for the ranking.

            You can adjust these weights to customize the ranking based on your specific needs.
            """)

        # Resume Upload with Drag & Drop
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader("Drag and drop resumes here", type=["docx", "pdf", "txt"], accept_multiple_files=True)

        # Process Resumes
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file not in st.session_state['uploaded_resumes']:  # Prevent reprocessing
                    filename = uploaded_file.name
                    file_extension = os.path.splitext(filename)[1].lower()
                    file_content = uploaded_file.read()

                    with st.spinner(f"Processing resume: {filename}"):
                        try:
                            skills, experience, preview_data, resume_text = process_resume(file_content, filename, file_extension[1:]) #Get tuple
                            if skills or experience:  # Only add if successfully processed
                                st.session_state['uploaded_resumes'].append((uploaded_file, preview_data, resume_text)) #Add for reference + Resume Text for Preview
                                st.success(f"Resume '{filename}' uploaded and processed successfully!")
                            else:
                                st.warning(f"Resume '{filename}' could not be fully processed.")

                        except Exception as e:
                            st.error(f"Error processing resume '{filename}': {e}")
                            logging.exception(f"Error processing resume '{filename}': {e}")

        # Show Uploaded Resumes (with preview)
        if st.session_state['uploaded_resumes']:
            st.subheader("Uploaded Resumes")
            for uploaded_file, preview_data, resume_text in st.session_state['uploaded_resumes']:
                with st.expander(f"Preview: {uploaded_file.name}"):  # Expander for Resume Preview
                    st.write(f"**Filename:** {uploaded_file.name}")
                    if preview_data:
                        st.write(f"**Skills:** {', '.join(preview_data['skills'])}") #Show skills
                        st.write(f"**Experience:** {preview_data['experience']} years") # Show Experience

                        # Show full resume Text
                        if resume_text:
                            st.subheader("Full Resume Text")
                            st.text_area("Resume Content", value=resume_text, height=200) #Text Preview

        # Job Description Input
        st.subheader("Job Description")
        job_description_file = st.file_uploader("Upload Job Description (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
        job_description_text = st.text_area("Or enter Job Description:", height=200)

        # Job Description Auto-Suggestions
        st.subheader("Job Description Auto-Suggestions")
        suggested_descriptions = ["Software Engineer Job Description", "Data Scientist Job Description", "Project Manager Job Description"] # Add more descriptions
        selected_suggestion = st.selectbox("Select a sample job description", suggested_descriptions)
        if selected_suggestion:
            st.write(f"Using Sample: {selected_suggestion}") # Show the selected suggestion
        # Ranking Weights
        st.subheader("Ranking Weights")
        skill_weight = st.slider("Skill Match Weight", 0, 20, SKILL_MATCH_WEIGHT)
        experience_weight = st.slider("Experience Weight", 0, 10, EXPERIENCE_WEIGHT)
        ml_weight = st.slider("ML Model Weight", 0, 10, ML_MODEL_WEIGHT)

    # Main Area
    st.header("Resume Ranking Results")

    # Job Description Processing
    job_keywords = []
    if job_description_file:
        try:
            file_extension = os.path.splitext(job_description_file.name)[1].lower()
            if file_extension == ".txt":
                job_description_text = job_description_file.read().decode("utf-8")
            elif file_extension == ".docx":
                job_description_text = parser.extract_text_from_docx(job_description_text)
            elif file_extension == ".pdf":
                job_description_text = parser.extract_text_from_pdf(job_description_text)

            job_keywords = extract_keywords_from_job_description(job_description_text)
            st.info("Extracted Job Keywords: " + ", ".join(job_keywords))

        except Exception as e:
            st.error(f"Error processing job description: {e}")
            job_keywords = []
    elif job_description_text:  #Use TextArea Job Description
        job_keywords = extract_keywords_from_job_description(job_description_text)
        st.info("Extracted Job Keywords: " + ", ".join(job_keywords))


    # Ranking
    if job_keywords and st.session_state['uploaded_resumes']: #Rank only when resumes are uploaded
        with Session() as session:
            resumes = session.query(Resume).all()

            for resume in resumes:
                resume.ranking_score = calculate_ranking_score(resume, job_keywords, skill_weight, experience_weight, ml_weight)

            session.commit()

        # Display ranked resumes using pandas DataFrame
        with Session() as session:
            resumes = session.query(Resume).all()

        data = []
        for resume in resumes:
            resume_skills = resume.skills.split(',') # Skills for each resume

            # Skill Matching Visualization
            fig = create_skill_matching_chart(resume_skills, job_keywords)
            st.plotly_chart(fig, use_container_width=True)

            data.append({
                "Filename": resume.filename,
                "Skills": resume.skills.split(','),
                "Experience": resume.experience,
                "Ranking Score": resume.ranking_score
            })

        if data:
            df = pd.DataFrame(data)
            df = df.sort_values(by="Ranking Score", ascending=False)

            #Add Sorting Options
            sort_by = st.selectbox("Sort By", ["Ranking Score", "Experience", "Filename"], index=0)
            ascending = st.checkbox("Ascending", False)
            df = df.sort_values(by=sort_by, ascending=ascending)

            st.dataframe(df) #Show Data
        else:
            st.info("No matching resumes found. Please check your job description and resume uploads.")
    elif not st.session_state['uploaded_resumes']:
        st.info("Please upload resumes to begin the ranking process.")
    elif not job_keywords:
        st.info("Please provide a job description to rank the resumes.")


if __name__ == "__main__":
    # Ensure the database tables exist
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        st.error(f"Database connection error: {e}. Please check your database settings.")

    # Ensure uploads directory exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
  
  
main()
