# models_ml/services/training_manager.py
import subprocess
import os
import datetime
import re
from huggingface_hub import login
from fastapi import HTTPException
# from pydantic import BaseModel
from fastapi.responses import JSONResponse
from models_ml.shared_models import TrainingRequest, HuggingFaceLoginRequest, ModelEntryResponse, RegisterModelRequest # ★★★ 이 줄을 추가합니다. ★★★
from models_ml.services import model_manager, data_manager

# training/train_data.py 스크립트의 경로를 지정합니다.
TRAIN_DATA_SCRIPT_PATH = "models_ml/training/train_data.py"

# main.py의 Pydantic 모델과 동일하게 정의
# class TrainingRequest(BaseModel):
#     model_id: str
#     system_message: str
#     load_in_4bit: bool = True
#     bnb_4bit_compute_dtype: str = 'bfloat16'
#     attn_implementation: str = 'eager'
#     lora_alpha: int = 128
#     lora_dropout: float = 0.05
#     lora_r: int = 64
#     lora_target_modules: str = 'all-linear'
#     per_device_train_batch_size: int = 1
#     gradient_accumulation_steps: int = 4
#     num_train_epochs: int = 10
#     learning_rate: float = 2e-4
#     lr_scheduler_type: str = 'constant'
#     optim: str = 'adamw_torch_fused'

# class HuggingFaceLoginRequest(BaseModel):
#     hf_token: str

# 모델 저장 디렉토리 (main.py와 동일)
OUTPUT_BASE_DIR = "models_ml/outputs"
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "adapters"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "merged"), exist_ok=True)

async def start_training(request_data: TrainingRequest):
    """
    학습 프로세스를 시작하고 로그를 반환합니다.
    """
    job_id = None
    try:
        data_file_path = data_manager.get_typed_file_path(request_data.file_type)

        if not os.path.exists(data_file_path):
            raise HTTPException(status_code=400, detail=f"학습 데이터 파일이 없습니다: {data_file_path}. 먼저 데이터를 업로드해주세요.")

        # 모델 저장 경로 동적으로 생성
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        job_id = f"job-{timestamp}"
        adapter_output_dir = os.path.join(OUTPUT_BASE_DIR, "adapters", job_id)
        merged_output_dir = os.path.join(OUTPUT_BASE_DIR, "merged", job_id)
        
        os.makedirs(adapter_output_dir, exist_ok=True)
        os.makedirs(merged_output_dir, exist_ok=True)

        # models_ml/training/train_data.py로 스크립트 경로 변경
        # 모든 학습 파라미터를 명령줄 인자로 전달
        command = [
            "python3",
            TRAIN_DATA_SCRIPT_PATH,
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
            "--per_device_train_batch_size", str(request_data.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(request_data.gradient_accumulation_steps),
            "--num_train_epochs", str(request_data.num_train_epochs),
            "--learning_rate", str(request_data.learning_rate),
            "--lr_scheduler_type", request_data.lr_scheduler_type,
            "--optim", request_data.optim,
        ]

        print(f"Executing training command: {' '.join(command)}")

        # subprocess.run을 사용하여 외부 스크립트 실행 (완료까지 대기)
        process = subprocess.run(
            command,
            capture_output=True, # 표준 출력 및 에러를 캡처
            text=True,           # 출력을 텍스트로 디코딩
            check=True           # 에러 발생 시 CalledProcessError 예외 발생
        )
        logs = process.stdout + process.stderr # 표준 출력과 에러를 모두 캡처

        # ★★★ 학습 완료 후 로그 파싱 및 DB 등록 ★★★
        eval_accuracy = None
        eval_loss = None

        # 로그에서 'eval_accuracy'와 'eval_loss' 값 추출
        # 정규 표현식을 사용하여 마지막 에포크의 평가 지표를 찾습니다.
        # eval_accuracy': 0.01764705963432789, 'eval_runtime': 0.4015, ... 'epoch': 1.0
        acc_match = re.search(r"'eval_accuracy':\s*([\d.]+)", logs)
        loss_match = re.search(r"'eval_loss':\s*([\d.]+)", logs)
        
        if acc_match:
            eval_accuracy = float(acc_match.group(1))
        if loss_match:
            eval_loss = float(loss_match.group(1))

        # DB에 모델 정보 등록
        model_manager.register_trained_model(
            RegisterModelRequest(
                job_id=job_id,
                base_model_id=request_data.model_id,
                adapter_path=adapter_output_dir,
                merged_path=merged_output_dir,
                eval_accuracy=eval_accuracy,
                eval_loss=eval_loss,
                lora_r=request_data.lora_r, # 사용된 lora_r 값 기록
                status='completed',
                description=f"학습 완료: {request_data.model_id} with LoRA r={request_data.lora_r}"
            )
        )
        print(f"모델 Job ID {job_id} 정보가 DB에 등록되었습니다.")
        # ★★★ 여기까지 추가 ★★★

        # 성공적으로 완료되면 출력 로그를 반환
        return {"status": "success", "message": "학습이 성공적으로 완료되었습니다.", "logs": process.stdout}
    except subprocess.CalledProcessError as e:
        # train_data.py 실행 중 오류 발생 시 에러 메시지 반환
        error_output = e.stdout + e.stderr
        print(f"Subprocess error: {error_output}")
        if job_id: # job_id가 생성된 경우에만 시도
             try:
                 model_manager.register_trained_model(
                     RegisterModelRequest(
                         job_id=job_id, # job_id를 사용하여 기존 레코드 업데이트 또는 새 레코드 추가
                         base_model_id=request_data.model_id,
                         adapter_path=os.path.join(OUTPUT_BASE_DIR, "adapters", job_id),
                         merged_path=os.path.join(OUTPUT_BASE_DIR, "merged", job_id),
                         status='failed',
                         description=f"학습 실패: {e.returncode}. {error_output[:100]}..."
                     )
                 )
                 print(f"모델 Job ID {job_id} 실패 정보가 DB에 등록되었습니다.")
             except Exception as db_e:
                 print(f"학습 실패 후 DB 등록 중 추가 오류: {db_e}")
        # 특정 에러 메시지를 파싱하여 사용자에게 더 친절한 메시지 제공
        if "out of memory" in error_output.lower() or "cuda out of memory" in error_output.lower():
            raise HTTPException(status_code=500, detail=f"학습 실패: GPU 메모리가 부족하거나, 파라미터가 너무 큽니다. {error_output}")
        elif "valueerror" in error_output.lower():
            raise HTTPException(status_code=500, detail=f"학습 실패: 설정값 오류 또는 데이터 문제. {error_output}")
        else:
            raise HTTPException(status_code=500, detail=f"학습 중 예기치 않은 오류 발생: {error_output}")
    except Exception as e:
        # 기타 예상치 못한 오류 처리
        print(f"Server error during training process: {e}")
        # 기타 예상치 못한 오류 발생 시 DB에 'failed' 상태로 등록
        if job_id:
             try:
                 model_manager.register_trained_model(
                     RegisterModelRequest(
                         job_id=job_id,
                         base_model_id=request_data.model_id,
                         adapter_path=os.path.join(OUTPUT_BASE_DIR, "adapters", job_id),
                         merged_path=os.path.join(OUTPUT_BASE_DIR, "merged", job_id),
                         status='failed',
                         description=f"서버 내부 오류: {str(e)[:100]}..."
                     )
                 )
                 print(f"모델 Job ID {job_id} 실패 정보가 DB에 등록되었습니다.")
             except Exception as db_e:
                 print(f"학습 실패 후 DB 등록 중 추가 오류: {db_e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

def huggingface_login(request_data: HuggingFaceLoginRequest):
    """
    Hugging Face 계정 로그인을 처리합니다.
    """
    print(f"HuggingFace 로그인 요청 수신: 토큰 길이 {len(request_data.hf_token)} ...") # NEW
    try:
        login(token=request_data.hf_token, add_to_git_credential=True)
        print("HuggingFace Hub 로그인 성공적으로 완료됨.") # NEW
        return {"status": "success", "message": "HuggingFace 로그인 성공."}
    except Exception as e:
        print(f"HuggingFace Hub 로그인 중 오류 발생: {e}") # NEW
        raise HTTPException(status_code=500, detail=f"HuggingFace 로그인 실패: {str(e)}")