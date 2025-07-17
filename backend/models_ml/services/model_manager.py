# models_ml/services/model_manager.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from fastapi import HTTPException
# from pydantic import BaseModel # 이 줄은 이제 필요 없습니다.
import datetime
from typing import Optional, List 
import os
import shutil

# ★★★ Pydantic 모델을 shared_models에서 임포트합니다. ★★★
from models_ml.shared_models import RegisterModelRequest, ModelEntryResponse, ModelActionRequest 
# ★★★ 사용되지 않는 임포트 제거: from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM

# PostgreSQL 데이터베이스 URL 설정
DATABASE_URL = "postgresql://myuser:mypassword@localhost/mydb" # ★★★ 여러분의 DB 정보로 변경 ★★★

# SQLAlchemy 엔진 및 세션 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# SQLAlchemy 모델 정의 (trained_models 테이블에 매핑)
class TrainedModelDB(Base):
    __tablename__ = "trained_models"

    job_id = Column(String, primary_key=True, index=True)
    base_model_id = Column(String, nullable=False)
    adapter_path = Column(String, nullable=False)
    merged_path = Column(String, nullable=False)
    training_date = Column(DateTime(timezone=True), default=datetime.datetime.now)
    eval_accuracy = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    lora_r = Column(Integer, nullable=True)
    status = Column(String, default='completed') # 'completed', 'failed', 'deployed', 'inactive'
    description = Column(Text, nullable=True)

# 데이터베이스 테이블 초기 생성
def create_db_tables():
    Base.metadata.create_all(bind=engine)
    print("PostgreSQL 'trained_models' 테이블이 생성되었거나 이미 존재합니다.")

# ★★★ Pydantic 모델 정의를 삭제합니다 (shared_models.py로 이동했음). ★★★
# class RegisterModelRequest(BaseModel): ... (삭제) ...
# class ModelEntryResponse(BaseModel): ... (삭제) ...
# class ModelActionRequest(BaseModel): ... (삭제) ...

# 1. 모델 등록 로직
def register_trained_model(model_data: RegisterModelRequest):
    db = SessionLocal()
    try:
        db_model = TrainedModelDB(**model_data.model_dump())
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        print(f"DB에 모델 '{db_model.job_id}' 등록 완료.")
        return db_model
    except Exception as e:
        db.rollback()
        print(f"모델 등록 중 DB 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"모델 등록 중 DB 오류 발생: {e}")
    finally:
        db.close()

# 2. 모든 모델 조회 로직
def get_all_models() -> List[TrainedModelDB]:
    db = SessionLocal()
    try:
        models = db.query(TrainedModelDB).all()
        return models
    except Exception as e:
        print(f"모델 조회 중 DB 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"모델 조회 중 DB 오류 발생: {e}")
    finally:
        db.close()

# 3. 모델 활성화(배포) 로직
def activate_model(job_id: str):
    db = SessionLocal()
    try:
        # 1. 현재 배포된 모든 모델을 'inactive'로 변경
        db.query(TrainedModelDB).filter(TrainedModelDB.status == 'deployed').update({TrainedModelDB.status: 'inactive'})

        # 2. 지정된 job_id 모델을 'deployed'로 변경
        model_to_deploy = db.query(TrainedModelDB).filter(TrainedModelDB.job_id == job_id).first()
        if not model_to_deploy:
            raise HTTPException(status_code=404, detail=f"Job ID '{job_id}'를 가진 모델을 찾을 수 없습니다.")
        
        model_to_deploy.status = 'deployed'
        db.commit()
        db.refresh(model_to_deploy)
        print(f"모델 '{job_id}'이(가) 성공적으로 배포되었습니다.")
        return {"status": "success", "message": f"모델 '{job_id}'이(가) 성공적으로 배포되었습니다."}
    except HTTPException as e:
        db.rollback()
        raise e
    except Exception as e:
        db.rollback()
        print(f"모델 배포 중 DB 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"모델 배포 중 오류 발생: {e}")
    finally:
        db.close()

# 4. 모델 삭제 로직
def delete_model(job_id: str):
    db = SessionLocal()
    try:
        model_to_delete = db.query(TrainedModelDB).filter(TrainedModelDB.job_id == job_id).first()
        if not model_to_delete:
            raise HTTPException(status_code=404, detail=f"Job ID '{job_id}'를 가진 모델을 찾을 수 없습니다.")
        
        # 물리적 파일 삭제
        if os.path.exists(model_to_delete.adapter_path):
            shutil.rmtree(model_to_delete.adapter_path)
            print(f"어댑터 폴더 삭제 완료: {model_to_delete.adapter_path}")
        if os.path.exists(model_to_delete.merged_path):
            shutil.rmtree(model_to_delete.merged_path)
            print(f"병합된 모델 폴더 삭제 완료: {model_to_delete.merged_path}")
        
        # DB에서 모델 정보 삭제
        db.delete(model_to_delete)
        db.commit()
        print(f"모델 '{job_id}'이(가) DB에서 성공적으로 삭제되었습니다.")
        return {"status": "success", "message": f"모델 '{job_id}'이(가) 성공적으로 삭제되었습니다."}
    except HTTPException as e:
        db.rollback()
        raise e
    except Exception as e:
        db.rollback()
        print(f"모델 삭제 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"모델 삭제 중 오류 발생: {e}")
    finally:
        db.close()