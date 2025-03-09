from sqlalchemy import create_engine, Column, Integer, String, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

Base = declarative_base()

class Resume(Base):
    __tablename__ = 'resumes'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    text_content = Column(Text)
    skills = Column(Text(4096))
    experience = Column(Integer)
    ranking_score = Column(Integer, default=0)
    ml_score = Column(Integer, default=0)

    __table_args__ = (
        Index('ix_filename', filename),
        Index('ix_ranking_score', ranking_score),
    )

    def __repr__(self):
        return f"<Resume(filename='{self.filename}', ranking_score={self.ranking_score})>"

# Database engine and session
print(f"DATABASE_URL: {DATABASE_URL}")  # Add this line
engine = create_engine(DATABASE_URL, echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
