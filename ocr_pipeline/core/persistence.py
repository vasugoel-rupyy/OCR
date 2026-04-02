import os
import json
import logging
import datetime
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("ocr_pipeline.persistence")

Base = declarative_base()

class OCRTaskResult(Base):
    __tablename__ = 'ocr_task_results'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(255), unique=True, index=True)
    request_id = Column(String(255), index=True)
    status = Column(String(50))
    document_type = Column(String(50))
    image_url = Column(Text)
    result_json = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class MySQLPersistence:
    def __init__(self):
        host = os.getenv("MYSQL_HOST", "mysql-db")
        user = os.getenv("MYSQL_USER", "ocruser")
        password = os.getenv("MYSQL_PASSWORD", "ocrpass")
        database = os.getenv("MYSQL_DATABASE", "ocrdb")
        
        # MySQL connection string using pymysql
        self.db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        self.engine = create_engine(
            self.db_url, 
            pool_size=10, 
            max_overflow=20,
            pool_recycle=3600
        )
        self.Session = sessionmaker(bind=self.engine)
        
        # Ensure table exists
        try:
            Base.metadata.create_all(self.engine)
            logger.info("MySQL persistence layer initialized and tables created.")
        except Exception as e:
            logger.error(f"Failed to initialize MySQL: {e}")

    def save_result(self, task_id: str, request_id: str, status: str, 
                    document_type: str, image_url: str, 
                    result: Optional[Dict[str, Any]] = None, 
                    error: Optional[str] = None):
        session = self.Session()
        try:
            # Check if exists (idempotency)
            existing = session.query(OCRTaskResult).filter_by(task_id=task_id).first()
            
            if existing:
                existing.status = status
                existing.result_json = result
                existing.error_message = error
            else:
                new_result = OCRTaskResult(
                    task_id=task_id,
                    request_id=request_id,
                    status=status,
                    document_type=document_type,
                    image_url=image_url,
                    result_json=result,
                    error_message=error
                )
                session.add(new_result)
            
            session.commit()
            logger.info(f"Saved task result {task_id} to MySQL.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save task result {task_id} to MySQL: {e}")
        finally:
            session.close()

# Singleton instance
persistence = MySQLPersistence()
