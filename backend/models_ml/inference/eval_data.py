# models_ml/inference/eval_data.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# bnb_4bit_compute_dtype을 인자로 받도록 수정합니다.
def run_inference(model_id: str, question: str, schema: str, bnb_4bit_compute_dtype: str = 'bfloat16') -> str:
    """
    주어진 모델 ID, 질문, 스키마, 그리고 compute_dtype을 사용하여 SQL 쿼리 또는 OA 답변을 추론합니다.
    """
    system_and_user_prompt = f"""You are an text to SQL query translator. Users will ask you questions and you will generate a SQL query based on the provided SCHEMA.

    SCHEMA:
    {schema}

    {question}"""

    full_prompt_string = f"<bos><start_of_turn>user\n{system_and_user_prompt}<end_of_turn>\n<start_of_turn>model\n"

    try:
        # 모델 로드 시 사용할 torch_dtype을 인자로부터 결정
        compute_dtype = eval(f"torch.{bnb_4bit_compute_dtype}")

        # 2. 토크나이저 및 모델 로드
        # trust_remote_code=True 추가
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=compute_dtype, # 인자로부터 받은 dtype 사용
            trust_remote_code=True # trust_remote_code 추가
        )
        model.eval()

        inputs = tokenizer(
            [full_prompt_string],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<start_of_turn>model\n" in generated_text:
            answer = generated_text.split("<start_of_turn>model\n")[-1].strip()
        else:
            answer = generated_text.strip()

        return answer

    except Exception as e:
        return f"모델 추론 중 오류 발생: {e}"
    finally:
        # 가비지 컬렉션 및 GPU 메모리 해제
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()

# 이 스크립트가 직접 실행될 때의 테스트 코드 (FastAPI에서 호출될 때는 실행되지 않습니다.)
if __name__ == '__main__':
    test_model_id = 'google/gemma-7b-it' # 또는 학습된 모델 경로
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
    result_sql = run_inference(test_model_id, test_question, test_schema, 'bfloat16') # compute_dtype 인자 추가
    print("\n▶ 추론 결과:")
    print(result_sql)