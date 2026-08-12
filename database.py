import os
from datetime import datetime
# pyrefly: ignore [missing-import]
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FallEvent(Base):
    __tablename__ = "fall_logs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(String, default="FALL")
    confidence = Column(Float)
    frames_analyzed = Column(Integer, default=60)
    media_type = Column(String)  # "image" or "video"
    media_path = Column(String)

class SetupSetting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    media_save_mode = Column(String, default="image") # "image" or "video"
    
# Create tables if not exist
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
