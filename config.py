

import os
import urllib.parse

# Heroku ClearDB database configuration
DATABASE_URL = os.environ.get("CLEARDB_DATABASE_URL")

if DATABASE_URL:
    # Parse the ClearDB URL
    urllib.parse.uses_netloc.append("mysql")
    url = urllib.parse.urlparse(DATABASE_URL)
    DATABASE_USER = url.username
    DATABASE_PASSWORD = url.password
    DATABASE_HOST = url.hostname
    DATABASE_NAME = url.path[1:]  # Remove leading slash

    DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}"
else:
    DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "12345")
    # Local development configuration
    DATABASE_USER = os.environ.get("DATABASE_USER", "student")  # Use dedicated user
    DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "12345") #Never keep this default password
    DATABASE_HOST = os.environ.get("DATABASE_HOST", "localhost")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "resume_db")
    DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}"


# Other configuration settings (e.g., upload folder)
UPLOAD_FOLDER = 'uploads'

# Ranking weights
SKILL_MATCH_WEIGHT = 10
EXPERIENCE_WEIGHT = 0
ML_MODEL_WEIGHT = 3  # Weight for machine learning model score

# File Paths
MODEL_PATH = os.path.join("ml_model", "model.joblib")
