import streamlit as st
import pandas as pd
from sqlalchemy.orm import sessionmaker
from models import engine, Resume
from resume_parser import parser
import os
import spacy
import logging
import joblib
import tempfile
from config import SKILL_MATCH_WEIGHT, EXPERIENCE_WEIGHT, ML_MODEL_WEIGHT, MODEL_PATH
import re  # Import the regular expression module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Database connection setup
Session = sessionmaker(bind=engine)

# Load the trained machine learning model
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Machine learning model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading machine learning model: {e}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'docx', 'pdf', 'txt'}


def process_resume(file_content, filename, file_type):
    """Extracts info, stores to database, and handles errors."""
    try:
        with Session() as session:
            if file_type == "docx":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                try:
                    text = parser.extract_text_from_docx(tmp_path)
                finally:
                    os.remove(tmp_path)  # Ensure cleanup

            elif file_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                try:
                    text = parser.extract_text_from_pdf(tmp_path)
                finally:
                    os.remove(tmp_path)

            elif file_type == "txt":
                text = file_content.decode("utf-8")
            else:
                return [], 0  # Unsupported file type

            if text:
                skills = parser.extract_skills_nlp(text)
                experience = parser.extract_experience_nlp(text)
                resume = Resume(filename=filename, text_content=text, skills=','.join(skills), experience=experience)
                session.add(resume)
                session.commit()
                return skills, experience
            else:
                return [], 0  # No text extracted

    except Exception as e:
        st.error(f"Error processing resume '{filename}': {e}")
        logging.exception(f"Error processing resume '{filename}': {e}")  # Log the full traceback
        return [], 0  # Consistent return on error


def extract_features(resume, job_keywords):
    """Extracts features for the machine learning model."""
    resume_skills = resume.skills.split(',')
    skill_match_count = sum(1 for skill in resume_skills if skill in job_keywords)
    experience = resume.experience
    return [skill_match_count, experience]


def calculate_ranking_score(resume, job_keywords):
    """Calculates ranking score based on skill match, experience, and ML."""
    resume_skills_str = resume.skills # Get the skills string from resume object
    resume_skills = resume_skills_str.split(',') if resume_skills_str else [] # Split string to list

    print(f"\n--- calculate_ranking_score for '{resume.filename}' ---") # Debug: Ranking function start
    print(f"Job Keywords: {job_keywords}")
    print(f"Resume Skills (raw): {resume.skills}") # Debug: Raw resume skills string
    print(f"Resume Skills (split): {resume_skills}") # Debug: Split resume skills list

    skill_match_count = 0
    matched_skills = [] # To track matched skills for debugging

    for resume_skill in resume_skills:
        for job_keyword in job_keywords:
            # Changed matching logic to substring and case-insensitive
            if job_keyword.strip().lower() in resume_skill.strip().lower(): # Substring and case-insensitive match
                skill_match_count += 1
                matched_skills.append(resume_skill) # Add to matched skills list
                logging.debug(f"Skill Match found: Resume Skill: '{resume_skill.strip().lower()}', Job Keyword: '{job_keyword.strip().lower()}'") # Debug log match
            else:
                logging.debug(f"No Skill Match: Resume Skill: '{resume_skill.strip().lower()}', Job Keyword: '{job_keyword.strip().lower()}'") # Debug log no match

    print(f"Skill Match Count: {skill_match_count}")
    print(f"Matched Skills: {matched_skills}") # Debug: Print matched skills
    total_job_keywords = len(job_keywords)
    skill_match_percentage = 0  # Default to 0%

    if total_job_keywords > 0:
        skill_match_percentage = (skill_match_count / total_job_keywords) * 100
        skill_match_percentage = round(skill_match_percentage, 1) #Round to 1 decimal place

    print(f"Skill Match Percentage: {skill_match_percentage}%") # Debug print percentage

    if model:
        features = extract_features(resume, job_keywords)
        try:
            ml_score = model.predict([features])[0]
            print(f"ML Score: {ml_score}") # Debug: ML Score
        except Exception as e:
            logging.warning(f"Error during model prediction: {e}")
            ml_score = 0
            print(f"ML Score Error: {ml_score}") # Debug: ML Error
    else:
        ml_score = 0
        print(f"ML Model Not Loaded: ML Score set to {ml_score}") # Debug: No ML Model

    score = min((skill_match_count * SKILL_MATCH_WEIGHT) + (resume.experience * EXPERIENCE_WEIGHT) + (ml_score * ML_MODEL_WEIGHT), 100)
    print(f"Final Ranking Score: {score}") # Debug: Final Score
    return score, skill_match_percentage # Return both scores


def extract_keywords_from_job_description(jd_text):
    """Extracts relevant keywords from the job description."""
    doc = nlp(jd_text)
    keywords = set()

    print("\n--- Tokens from Job Description (Detailed Debug) ---")  # Debug print
    for token in doc:
        print(f"Token: '{token.text}'")
        print(f"  POS: {token.pos_}, is_stop: {token.is_stop}, is_punct: {token.is_punct}, is_currency: {token.is_currency}, is_digit: {token.is_digit}, like_num: {token.like_num}")
        is_pos_ok = token.pos_ in ["NOUN", "ADJ", "PROPN", "VERB"]
        is_stop_ok = not token.is_stop
        is_punct_ok = not token.is_punct and not token.is_currency and not token.is_digit
        is_range_ok = not re.match(r'^\d+-\d+$', token.text) and not token.like_num

        print(f"  POS Check: {is_pos_ok}, Stop Word Check: {is_stop_ok}, Punct/Symbol Check: {is_punct_ok}, Range Check: {is_range_ok}")


        if is_pos_ok and is_stop_ok and is_punct_ok and is_range_ok:
            keywords.add(token.lemma_.lower())
            print(f"  ==> ADDED Keyword: '{token.text}' (lemma: '{token.lemma_}')") # Indicate when added
        else:
            print("  ==> FILTERED OUT") # Indicate when filtered

    print("\n--- Final Extracted Keywords ---")
    print(keywords)
    return list(keywords)



def main():
    st.title("AI-Powered Resume Screening and Ranking System")

    print("\n--- main() function execution START ---") # Debug: Track main execution

    # --- Sidebar (Optional) ---
    with st.sidebar:
        st.subheader("Select Theme")
        theme = st.radio("Choose a theme", ["Light", "Dark"])  # Not used, but an example

        st.subheader("Inputs and Configuration")
        with st.expander("Ranking Weights Explained"):
            st.write(f"Skill Match Weight: {SKILL_MATCH_WEIGHT}")
            st.write(f"Experience Weight: {EXPERIENCE_WEIGHT}")
            st.write(f"ML Model Weight: {ML_MODEL_WEIGHT}")

    # --- Resume Upload Section ---
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader("Drag and drop files here", type=["docx", "pdf", "txt"], accept_multiple_files=True,
                                        help="Limit 200MB per file • TXT, DOCX, PDF")

    if uploaded_files:
        print("Resumes Uploaded: Processing...") # Debug: Resume upload detected
        if 'uploaded_resumes' not in st.session_state:
            st.session_state['uploaded_resumes'] = []

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            file_extension = os.path.splitext(filename)[1].lower()
            file_content = uploaded_file.read()

            with st.spinner(f"Processing resume: {filename}"):
                skills, experience = process_resume(file_content, filename, file_extension[1:])
                if skills or experience: # Check if processed
                    st.success(f"Resume '{filename}' uploaded and processed successfully!")
                    st.session_state['uploaded_resumes'].append(filename)  # Store filename
                #No need of else, process_resume already handles the errors

    # --- Job Description Upload ---
    st.subheader("Job Description")
    job_description_file = st.file_uploader("Upload Job Description (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])

    if job_description_file:
        print("Job Description Uploaded: Processing...") # Debug: JD upload detected
        try:
            file_extension = os.path.splitext(job_description_file.name)[1].lower()
            if file_extension == ".txt":
                job_description_text = job_description_file.read().decode("utf-8")
            elif file_extension == ".docx":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(job_description_file.read())
                    tmp_path = tmp.name
                try:
                    job_description_text = parser.extract_text_from_docx(tmp_path)
                finally:
                    os.remove(tmp_path)
            elif file_extension == ".pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(job_description_file.read())
                    tmp_path = tmp.name
                try:
                    job_description_text = parser.extract_text_from_pdf(tmp_path)
                finally:
                    os.remove(tmp_path)

            print("\n--- Job Description Text (Before Keyword Extraction) ---") # Debug print: JD Text
            print(f"First 200 chars of JD Text: {job_description_text[:200]}")

            job_keywords = extract_keywords_from_job_description(job_description_text) # CALL KEYWORD EXTRACTION FUNCTION
            st.write("Extracted Job Keywords:", job_keywords) # DISPLAY THE *KEYWORDS* - NOT RAW TEXT

        except Exception as e:
            st.error(f"Error processing job description: {e}")
            logging.exception(f"Error processing job description: {e}")
            job_keywords = []  # Initialize to empty list on error

        print("\n--- Job Keywords Extracted and Displayed ---") # Debug: Keywords Displayed

        # --- Ranking ---
        st.header("Resume Ranking Results")

        if job_keywords and 'uploaded_resumes' in st.session_state and st.session_state['uploaded_resumes']:
            print("Ranking Resumes...") # Debug: Ranking process started
            print(f"Job Keywords for Ranking: {job_keywords}") # Debug: Job Keywords for Ranking
            print(f"Uploaded Resumes for Ranking: {st.session_state['uploaded_resumes']}") # Debug: Uploaded Resumes

            with Session() as session:
                # Efficiently fetch only the resumes we need
                ranked_resumes = session.query(Resume).filter(Resume.filename.in_(st.session_state['uploaded_resumes'])).all() # Renamed to ranked_resumes

                print(f"\n--- Resumes Fetched for Ranking ---") # Debug: Resumes Fetched
                for resume in ranked_resumes:
                    print(f"  Fetched Resume: {resume.filename}, Skills: {resume.skills}, Experience: {resume.experience}")

                data = []
                print(f"\n--- Creating DataFrame for Ranking ---") # Debug: DataFrame Creation
                for resume in ranked_resumes:
                    ranking_score, skill_match_percentage = calculate_ranking_score(resume, job_keywords) # Get both scores
                    print(f"  Resume for DataFrame: {resume.filename}, Ranking Score: {ranking_score}, Skill Match Percentage: {skill_match_percentage}%, Skills: {resume.skills}, Experience: {resume.experience}") # Debug: Data for DataFrame
                    data.append({
                        "Filename": resume.filename,
                        "Skills": resume.skills.split(',') if resume.skills else [], #Handle empty
                        "Experience": resume.experience,
                        "Ranking Score": ranking_score,
                        "Skill Match Percentage": skill_match_percentage # New column
                    })

                df = pd.DataFrame(data)
                df = df.sort_values(by="Ranking Score", ascending=False)
                st.dataframe(df)
                print("Ranking Table Displayed.") # Debug: Ranking table displayed
        elif not job_keywords:
            st.write("Please upload a job description to begin the ranking process.")
            print("No Job Description Message Displayed.") # Debug: No JD message
        elif 'uploaded_resumes' not in st.session_state or not st.session_state['uploaded_resumes']:
            st.write("Please upload resumes to begin the ranking process.")
            print("No Resumes Message Displayed (Initial).") # Debug: No resumes message
        else:
            st.write("No resumes to rank.")
            print("No Resumes to Rank Message Displayed.") # Debug: No resumes to rank message

    print("\n--- main() function execution END ---") # Debug: Track main execution


if __name__ == "__main__":
    from models import Base, engine  # Local import to avoid circular dependency
    Base.metadata.create_all(engine)  # Create tables if they don't exist
    if not os.path.exists("uploads"):  # You don't actually need this with tempfile
        os.makedirs("uploads")
    main()