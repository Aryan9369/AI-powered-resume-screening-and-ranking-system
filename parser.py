import docx
import spacy
import re
import logging
from pdfminer.high_level import extract_text

# Configure logging (if not already configured elsewhere)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_docx(docx_path):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(docx_path)
        full_text = '\n'.join([para.text for para in doc.paragraphs])
        logging.info(f"Successfully extracted text from DOCX: {docx_path[:50]}...")  # Log success
        return full_text
    except Exception as e:
        logging.error(f"Error extracting text from DOCX {docx_path}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        logging.info(f"Successfully extracted text from PDF: {pdf_path[:50]}...") # Log success
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

def extract_skills_nlp(text):
    """Extracts skills from text using spaCy NLP and keyword matching, handling spaces and commas."""
    logging.info("Starting skill extraction (handling spaces and commas)...") # Debug log
    if not text:
        logging.warning("No text provided for skill extraction.") # Debug log if no text
        return []

    doc = nlp(text)
    skills = set()
    keywords = ["skill", "skills", "expert", "proficiency", "expertise", "knowledge", "proficient"]
    # Expanded skill list - Add more skills relevant to your domain!
    skill_list = [
        "Python", "Java", "JavaScript", "C++", "C#", "SQL", "NoSQL", "Machine Learning", "Deep Learning",
        "Data Analysis", "Data Mining", "Data Warehousing", "ETL", "Data Visualization", "Tableau", "Power BI",
        "Communication", "Project Management", "Agile", "Scrum", "Leadership", "Teamwork", "Problem-solving",
        "Analytical Skills", "Critical Thinking", "Time Management", "Cloud Computing", "AWS", "Azure", "GCP",
        "Docker", "Kubernetes", "REST APIs", "Web Services", "Software Development", "Testing", "Debugging",
        "Git", "Version Control", "Databases", "Algorithms", "Data Structures", "Statistical Modeling", "NLP", "Computer Vision",
        "Linux", "Windows", "Networking", "Cybersecurity", "Frontend Development", "Backend Development", "Mobile Development",
        "React", "Angular", "Vue.js", "Node.js", "Spring Boot", ".NET", "TensorFlow", "PyTorch", "Scikit-learn", "Streamlit", "Flask", "Django", "CSS", "HTML", "JavaScript", "TypeScript"
    ]

    logging.info(f"Number of tokens in processed text for skills: {len(doc)}") # Debug log token count

    # Split text into potential skill phrases by commas and newlines, and then further by spaces
    potential_skill_phrases = []
    for phrase_delimiter in ['\n', ',']: # Split by newlines and commas first
        phrases = text.split(phrase_delimiter)
        for phrase in phrases:
            potential_skill_phrases.extend(phrase.strip().split()) # Further split by spaces and strip whitespace

    for phrase in potential_skill_phrases:
        doc_phrase = nlp(phrase) # Process each potential skill phrase with spaCy
        for token in doc_phrase:
            logging.debug(f"Token for skill extraction: '{token.text}', POS: {token.pos_}") # Debug each token
            if token.pos_ in ["NOUN", "ADJ"] and token.text in skill_list:
                skills.add(token.text)
                logging.debug(f"Skill added: '{token.text}'") # Debug log when skill is added

    for ent in doc.ents:
        logging.debug(f"Entity for skill extraction: '{ent.text}', Label: {ent.label_}") # Debug each entity
        if ent.label_ == "ORG" and any(keyword in ent.text.lower() for keyword in keywords):
            skills.add(ent.text)
            logging.debug(f"Skill added from entity: '{ent.text}'") # Debug log when skill added from entity

    final_skills = list(skills)
    logging.info(f"Final extracted skills: {final_skills}") # Debug log final skills
    return final_skills


def extract_experience_nlp(text):
    """Extracts experience (years) from text using NLP and regex."""
    logging.info("Starting experience extraction...") # Debug log
    if not text:
        logging.warning("No text provided for experience extraction.") # Debug log if no text
        return 0

    doc = nlp(text)
    years = 0

    logging.info(f"Number of entities in processed text for experience: {len(doc.ents)}") # Debug log entity count

    for ent in doc.ents:
        logging.debug(f"Entity for experience extraction: '{ent.text}', Label: {ent.label_}") # Debug each entity
        if ent.label_ == "DATE":
            match = re.search(r"(\d+)\s*years", ent.text, re.IGNORECASE)
            if match:
                years += int(match.group(1))
                logging.debug(f"Experience found in date entity: '{ent.text}', years: {match.group(1)}") # Debug log experience from entity

    matches = re.findall(r"(\d+)\+?\s*years", text, re.IGNORECASE)
    for match in matches:
        try:
            years += int(match)
            logging.debug(f"Experience found by regex: '{match}', total years: {years}") # Debug log experience from regex
        except ValueError:
            pass

    logging.info(f"Total experience extracted: {years} years") # Debug log total experience
    return years