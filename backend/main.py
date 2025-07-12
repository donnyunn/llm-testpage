#main.py
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import shutil
from huggingface_hub import login
from models_ml.eval_data import run_inference
import pandas as pd
import json
import gc
import datetime

app = FastAPI()

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

class TrainingRequest(BaseModel):
    model_id: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    system_message: str

class TrainingRequest(BaseModel):
    model_id: str
    system_message: str
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = 'bfloat16'
    attn_implementation: str = 'eager'
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_r: int = 64
    lora_target_modules: str = 'all-linear'
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 10
    learning_rate: float = 2e-4
    lr_scheduler_type: str = 'constant'
    optim: str = 'adamw_torch_fused'

class InferenceRequest(BaseModel):
    model_id: str
    question: str
    schema_info: str

class DataEntry(BaseModel):
    id: int
    question: str
    answer: str
    schema: str

class NewDataEntry(BaseModel):
    question: str
    answer: str
    schema: str

class DeleteRequest(BaseModel):
    id: int

UPLOAD_FOLDER = "models_ml/data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
FILE_PATH = os.path.join(UPLOAD_FOLDER, "uploaded_data.xlsx")

OUTPUT_BASE_DIR = "models_ml/outputs"
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "adapters"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "merged"), exist_ok=True)

def read_excel_file():
    if not os.path.exists(FILE_PATH):
        return pd.DataFrame(columns=['id', 'question', 'answer', 'schema'])
    try:
        df = pd.read_excel(FILE_PATH)
        df = df.fillna('')
        if 'id' in df.columns:
            df['id'] = df['id'].astype(int)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return pd.DataFrame(columns=['id', 'question', 'answer', 'schema'])

def write_excel_file(df):
    try:
        df.to_excel(FILE_PATH, index=False)
        return True
    except Exception as e:
        print(f"Error writing to Excel file: {e}")
        return False

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        with open(FILE_PATH, "wb") as f:
            f.write(await file.read())
        return {"status": "success", "message": "파일이 성공적으로 업로드되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류 발생: {e}")

@app.get("/data-entries")
async def get_data_entries():
    data = read_excel_file()
    return {"status": "success", "data": data.to_dict(orient='records')}

@app.post("/add-data")
async def add_data(entry: NewDataEntry):
    df = read_excel_file()
    new_id = df['id'].max() + 1 if not df.empty and 'id' in df.columns else 1
    
    new_row = pd.DataFrame([{'id': new_id, **entry.model_dump()}])
    
    updated_df = pd.concat([df, new_row], ignore_index=True)

    if write_excel_file(updated_df):
        return {"status": "success", "message": "새로운 데이터가 성공적으로 추가되었습니다."}
    else:
        raise HTTPException(status_code=500, detail="파일 저장 중 오류가 발생했습니다.")

@app.post("/update-data")
async def update_data(entry: DataEntry):
    df = read_excel_file()

    if not df.empty and entry.id in df['id'].values:
        df.loc[df['id'] == entry.id, 'question'] = entry.question
        df.loc[df['id'] == entry.id, 'answer'] = entry.answer
        df.loc[df['id'] == entry.id, 'schema'] = entry.schema

        if write_excel_file(df):
            return {"status": "success", "message": "데이터가 성공적으로 업데이트되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="파일 저장 중 오류가 발생했습니다.")
    else:
        raise HTTPException(status_code=404, detail=f"업데이트할 데이터를 찾을 수 없습니다. (ID: {entry.id})")

@app.post("/delete-data")
async def delete_data(request: DeleteRequest):
    df = read_excel_file()

    if not df.empty and request.id in df['id'].values:
        df = df[df['id'] != request.id]

        if write_excel_file(df):
            return {"status": "success", "message": "데이터가 성공적으로 삭제되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="파일 저장 중 오류가 발생했습니다.")
    else:
        raise HTTPException(status_code=404, detail=f"삭제할 데이터를 찾을 수 없습니다. (ID: {request.id})")

@app.post("/start_training_test")
async def start_training_test(request_data: TrainingRequest):
    try:
        data_file_path = "models_ml/data/uploaded_data.xlsx"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        job_id = f"job-{timestamp}"
        adapter_output_dir = os.path.join(OUTPUT_BASE_DIR, "adapters", job_id)
        merged_output_dir = os.path.join(OUTPUT_BASE_DIR, "merged", job_id)

        command = [
            "python3", 
            "models_ml/train_data.py",
            "--model_id", request_data.model_id,
            "--system_message", request_data.system_message,
            "--adapter_output_dir", adapter_output_dir, 
            "--merged_output_dir", merged_output_dir,   
            "--data_file_path", data_file_path,
            "--load_in_4bit", str(request_data.load_in_4bit),
            "--bnb_4bit_compute_dtype", request_data.bnb_4bit_compute_dtype,
            "--attn_implementation", request_data.attn_implementation,
            "--lora_alpha", str(request_data.lora_alpha),
            "--lora_dropout", str(request_data.lora_dropout),
            "--lora_r", str(request_data.lora_r),
            "--lora_target_modules", request_data.lora_target_modules,
            "--per_device_train_batch_size", str(request_data.train_batch_size),
            "--gradient_accumulation_steps", str(request_data.gradient_accumulation_steps),
            "--num_train_epochs", str(request_data.num_train_epochs),
            "--learning_rate", str(request_data.learning_rate),
            "--lr_scheduler_type", request_data.lr_scheduler_type,
            "--optim", request_data.optim,
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return {"status": "success", "message": "학습 완료.", "logs": process.stdout}
    except subprocess.CalledProcessError as e:
        error_output = e.stdout + e.stderr
        print(f"Subprocess error: {error_output}")

        if "out of memory" in error_output.lower():
            raise HTTPException(status_code=500, detail=f"학습 실패: 메모리가 부족합니다. {error_output}")
        else:
            raise HTTPException(status_code=500, detail=f"학습 중 오류 발생: {error_output}")
    except Exception as e:
        print(f"Server error: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")

@app.post("/run_inference")
async def execute_inference(request_data: InferenceRequest):
    try:
        predicted_sql = run_inference(
            model_id=request_data.model_id,
            question=request_data.question,
            schema=request_data.schema_info
        )
        return {"status": "success", "predicted_sql": predicted_sql}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

class HuggingFaceLoginRequest(BaseModel):
    hf_token: str

@app.post("/huggingface/login")
async def huggingface_login(request: HuggingFaceLoginRequest):
    try:
        login(token=request.hf_token, add_to_git_credential=True)
        return {"status": "success", "message": "HuggingFace 로그인 성공."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace 로그인 실패: {e}")
