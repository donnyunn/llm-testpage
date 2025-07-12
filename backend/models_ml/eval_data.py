import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def run_inference(model_id: str, question: str, schema: str) -> str:
    """
    주어진 모델 ID, 질문, 스키마를 사용하여 SQL 쿼리 또는 OA 답변을 추론합니다.
    """
    # 1. 메시지 형식 구성: Gemma 모델이 이해하는 대화 포맷을 따릅니다.
    #    System message와 User message 내용을 하나의 User Prompt로 결합
    system_and_user_prompt = f"""You are an text to SQL query translator. Users will ask you questions and you will generate a SQL query based on the provided SCHEMA.

    SCHEMA:
    {schema}

    {question}"""

    # Gemma Instruction-tuned 모델의 대화 포맷
    # <bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n
    full_prompt_string = f"<bos><start_of_turn>user\n{system_and_user_prompt}<end_of_turn>\n<start_of_turn>model\n"

    try:
        # 2. 토크나이저 및 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # CPU 환경에서 large 모델 로딩 시 메모리 부족 또는 매우 느림.
        # GPU 환경이라면 device_map="auto", torch_dtype=torch.float16/bfloat16 사용 권장.
        # CPU에서 테스트 시, 모델 크기가 너무 크면 로드 자체가 실패할 수 있습니다.
        # 이 경우 더 작은 모델(예: gemma-2b-it)을 시도하거나,
        # torch_dtype을 torch.float32로 변경하고 CPU 메모리를 충분히 확보해야 합니다.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 # GPU 환경 권장. CPU only라면 float32로 변경 고려.
        )
        model.eval() # 평가 모드로 전환 (Dropout 등 비활성화)

        # 3. 입력 텍스트를 모델 입력 형식(토큰 ID)으로 변환
        inputs = tokenizer(
            [full_prompt_string], # 입력을 리스트 형태로 전달 (배치 처리 가능)
            return_tensors="pt",  # PyTorch 텐서로 반환
            padding=True,         # 배치 내에서 가장 긴 시퀀스에 맞춰 패딩
            truncation=True       # 최대 토큰 길이를 초과하면 잘라냄
        ).to(model.device) # 모델과 동일한 디바이스(CPU/GPU)로 텐서 이동

        # 4. 모델 추론 (텍스트 생성)
        outputs = model.generate(
            **inputs,             # 입력 텐서 (input_ids, attention_mask 등)
            max_new_tokens=256,   # 생성할 최대 토큰 수
            do_sample=False,      # 샘플링 비활성화 (가장 확률 높은 다음 토큰 선택, 안정적)
            pad_token_id=tokenizer.eos_token_id # 패딩 토큰을 EOS 토큰으로 설정 (경고 감소)
        )

        # 5. 생성된 토큰을 다시 텍스트로 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 6. ChatML 포맷에서 모델의 최종 답변만 추출
        if "<start_of_turn>model\n" in generated_text:
            # "<start_of_turn>model\n" 이후의 내용을 답변으로 간주
            answer = generated_text.split("<start_of_turn>model\n")[-1].strip()
        else:
            # 예상치 못한 형식일 경우 전체 텍스트를 답변으로 간주
            answer = generated_text.strip()

        return answer

    except Exception as e:
        # 오류 발생 시 에러 메시지 반환
        return f"모델 추론 중 오류 발생: {e}"
    finally:
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()

# 이 스크립트가 직접 실행될 때의 테스트 코드 (FastAPI에서 호출될 때는 실행되지 않습니다.)
if __name__ == '__main__':
    test_model_id = 'google/gemma-7b-it'
    test_question = "모든 직원의 이름과 부서를 보여줘"
    test_schema = """
    CREATE TABLE Employees (
        employee_id INT PRIMARY KEY,
        name VARCHAR(255),
        department VARCHAR(255),
        salary INT
    );
    CREATE TABLE Sales (
        sale_id INT PRIMARY KEY,
        employee_id INT,
        product VARCHAR(255),
        amount INT,
        sale_date DATE
    );
    """
    print(f"테스트 시작: 모델 ID '{test_model_id}'")
    result_sql = run_inference(test_model_id, test_question, test_schema)
    print("\n▶ 추론 결과:")
    print(result_sql)
