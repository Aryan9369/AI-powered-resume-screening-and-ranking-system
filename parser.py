import docx
import spacy
import re
import logging
from pdfminer.high_level import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_docx(docx_path):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(docx_path)
        full_text = '\n'.join([para.text for para in doc.paragraphs])
        return full_text
    except Exception as e:
        logging.error(f"Error extracting text from {docx_path}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_skills_nlp(text):
    """Extracts skills from text using spaCy NLP and keyword matching."""
    doc = nlp(text)
    skills = set()

    # Skill keywords
    keywords = ["skill", "skills", "expert", "proficiency", "expertise", "knowledge", "proficient"]

    # Example skill list (replace with a proper skill ontology)
    skill_list = ["Python", "Java", "SQL", "Machine Learning", "Data Analysis", "Communication", "Project Management"]  # Extend this significantly

    # Extract skills based on noun/adj and keyword matching
    for token in doc:
        if token.pos_ in ["NOUN", "ADJ"] and token.text in skill_list:  # Check against skill list
            skills.add(token.text)

    # If the skill is in keyword form then add it
    for ent in doc.ents:
        if ent.label_ == "ORG" and any(keyword in ent.text.lower() for keyword in keywords):
            skills.add(ent.text)
    return list(skills)

def extract_experience_nlp(text):
    """Extracts experience (years) from text using NLP and regex."""
    doc = nlp(text)
    years = 0
    for ent in doc.ents:
        if ent.label_ == "DATE":
            # Example: "5 years of experience"
            match = re.search(r"(\d+)\s*years", ent.text, re.IGNORECASE)
            if match:
                years += int(match.group(1))

    # More robust regex to find experience described differently
    matches = re.findall(r"(\d+)\+?\s*years", text, re.IGNORECASE)
    for match in matches:
        try:
            years += int(match)
        except ValueError:
            pass

    return years
