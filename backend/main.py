#main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import subprocess
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware

# 모든 Pydantic 모델을 shared_models에서 임포트합니다.
from models_ml.shared_models import (
    TrainingRequest, InferenceRequest, DataEntry, NewDataEntry, 
    DeleteRequest, HuggingFaceLoginRequest, ModelActionRequest
)

# 서비스 매니저들 임포트
from models_ml.services import training_manager, data_manager, inference_manager, model_manager 

app = FastAPI(on_startup=[model_manager.create_db_tables])

origins = [
    "http://localhost:5173",
    f"http://{os.environ.get('UBUNTU_PC_IP', '192.168.0.43')}:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-text-to-sql-data") # Text-to-SQL용 업로드
async def upload_text_to_sql_data(file: UploadFile = File(...)):
    return await data_manager.upload_typed_data(file, "text-to-sql")

@app.post("/upload-oa-qna-data") # OA Q&A용 업로드
async def upload_oa_qna_data(file: UploadFile = File(...)):
    return await data_manager.upload_typed_data(file, "oa-qna")

# @app.post("/upload-data")
# async def upload_data(file: UploadFile = File(...)):
#     return await data_manager.upload_data(file)

@app.get("/data-entries")
async def get_data_entries(file_type: str = Query(..., description="데이터의 유형 (text-to-sql 또는 oa-qna)")):
    return data_manager.get_data_entries(file_type)

@app.post("/add-data/{file_type}")
async def add_data(file_type: str, entry: BaseModel): # entry는 NewDataEntry 또는 NewOAQnAEntry가 될 수 있도록 BaseModel로 받음
    if file_type == "text-to-sql":
        typed_entry = NewDataEntry(**entry.model_dump())
    elif file_type == "oa-qna":
        typed_entry = NewOAQnAEntry(**entry.model_dump())
    else:
        raise HTTPException(status_code=400, detail="유효하지 않은 파일 유형입니다.")
    return data_manager.add_data(typed_entry, file_type)

@app.post("/update-data/{file_type}")
async def update_data(file_type: str, entry: DataEntry): # DataEntry는 id를 포함해야 함
    return data_manager.update_data(entry, file_type)

@app.post("/delete-data/{file_type}")
async def delete_data(file_type: str, request: DeleteRequest):
    return data_manager.delete_data(request, file_type)

@app.post("/start_training_test")
async def start_training_test(request_data: TrainingRequest):
    return await training_manager.start_training(request_data)

@app.post("/run_inference")
async def execute_inference(request_data: InferenceRequest):
    return inference_manager.get_inference_result(request_data)

@app.post("/huggingface/login")
async def huggingface_login(request: HuggingFaceLoginRequest):
    return training_manager.huggingface_login(request)

@app.get("/api/models")
async def get_models_list():
    models = model_manager.get_all_models()
    return {"status": "success", "data": [model_manager.ModelEntryResponse.model_validate(model) for model in models]}

@app.post("/api/models/activate")
async def activate_model_api(request: ModelActionRequest):
    return model_manager.activate_model(request.job_id)

@app.post("/api/models/delete")
async def delete_model_api(request: ModelActionRequest):
    return model_manager.delete_model(request.job_id)