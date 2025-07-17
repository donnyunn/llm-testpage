# models_ml/services/inference_manager.py
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional # Optional 임포트

# 실제 추론 로직이 있는 eval_data.py의 run_inference 함수 임포트
from models_ml.inference.eval_data import run_inference

# main.py의 Pydantic 모델과 동일하게 정의 (bnb_4bit_compute_dtype 추가)
class InferenceRequest(BaseModel):
    model_id: str
    question: str
    schema_info: Optional[str] = None
    bnb_4bit_compute_dtype: str = 'bfloat16' # ★★★ 이 줄을 추가합니다. ★★★

def get_inference_result(request_data: InferenceRequest):
    try:
        predicted_sql = run_inference(
            model_id=request_data.model_id,
            question=request_data.question,
            schema=request_data.schema_info,
            bnb_4bit_compute_dtype=request_data.bnb_4bit_compute_dtype # ★★★ 이 인자 전달 ★★★
        )
        return {"status": "success", "predicted_sql": predicted_sql}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {str(e)}")