# models_ml/shared_models.py
from pydantic import BaseModel
from typing import Optional
import datetime # ★★★ 이 줄이 있는지 반드시 확인해주세요. ★★★

# 모든 Pydantic 모델을 여기에 정의합니다.
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
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 10
    learning_rate: float = 2e-4
    lr_scheduler_type: str = 'constant'
    optim: str = 'adamw_torch_fused'
    file_type: str

class InferenceRequest(BaseModel):
    model_id: str
    question: str
    schema_info: Optional[str] = None
    bnb_4bit_compute_dtype: str = 'bfloat16'

class DataEntry(BaseModel):
    id: int
    question: str
    answer: str
    schema: str

class NewDataEntry(BaseModel):
    question: str
    answer: str
    schema: str

class NewOAQnAEntry(BaseModel): # OA Q&A용 Pydantic 모델 (schema 없음)
    question: str
    answer: str

class DeleteRequest(BaseModel):
    id: int

class HuggingFaceLoginRequest(BaseModel):
    hf_token: str

class RegisterModelRequest(BaseModel):
    job_id: str
    base_model_id: str
    adapter_path: str
    merged_path: str
    eval_accuracy: Optional[float] = None
    eval_loss: Optional[float] = None
    lora_r: Optional[int] = None
    status: str = 'completed'
    description: Optional[str] = None

class ModelActionRequest(BaseModel):
    job_id: str

# model_manager에서 사용할 응답 Pydantic 모델도 이곳에 정의
class ModelEntryResponse(BaseModel):
    job_id: str
    base_model_id: str
    adapter_path: str
    merged_path: str
    training_date: datetime.datetime
    eval_accuracy: Optional[float] = None
    eval_loss: Optional[float] = None
    lora_r: Optional[int] = None
    status: str
    description: Optional[str] = None

    class Config:
        from_attributes = True
