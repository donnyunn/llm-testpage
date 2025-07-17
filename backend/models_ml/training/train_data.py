# models_ml/train_data.py
import json
import pandas as pd
import numpy as np
import torch
import os
import argparse
from datasets import Dataset, load_dataset
# Hugging Face 토큰은 캐시된 것을 사용하므로 login 함수는 필요 없습니다.
# from huggingface_hub import login
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, setup_chat_format

def generate_messages(data, system_message_template): # system_message_template 인자 추가
    """
    주어진 데이터를 모델 학습을 위한 대화 형식으로 변환합니다.
    """
    # system_message_template을 사용하여 동적으로 시스템 메시지 생성
    system_message = system_message_template.format(schema=data["schema"]) # data["context"] 대신 data["schema"] 사용
    return {
        "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": data["question"]},
        {"role": "assistant", "content": data["answer"]}
        ]
    }

def compute_metrics(eval_pred):
    """
    평가 단계에서 모델의 정확도를 계산합니다.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).astype(np.float32)
    accuracy = correct.mean().item()
    return {"accuracy": accuracy}

def train():
    """
    학습 파이프라인의 주요 로직을 담고 있는 함수입니다.
    """
    # 1. 커맨드라인 인자 파서 설정
    parser = argparse.ArgumentParser(description="PEFT Model Training Script")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face Model ID to use as base.")
    parser.add_argument("--data_file_path", type=str, required=True,
                        help="Path to the training data file (xlsx or jsonl).")
    parser.add_argument("--adapter_output_dir", type=str, required=True,
                        help="Directory to save the PEFT adapter.")
    parser.add_argument("--merged_output_dir", type=str, required=True,
                        help="Directory to save the merged model.")
    parser.add_argument("--system_message", type=str, required=True,
                        help="System message template for the model.")
    
    # 학습 파라미터 인자 추가
    parser.add_argument("--load_in_4bit", type=lambda x: x.lower() == 'true', default=True, # bool 타입 파싱을 위해 lambda 사용
                        help="Whether to load the model in 4-bit precision.")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default='bfloat16',
                        help="The compute dtype for 4-bit quantization (e.g., bfloat16, float16).")
    parser.add_argument("--attn_implementation", type=str, default='eager',
                        help="Attention implementation to use (eager or flash_attention_2).")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout parameter.")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA r (rank) parameter.")
    parser.add_argument("--lora_target_modules", type=str, default='all-linear',
                        help="LoRA target modules (comma-separated if multiple).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default='constant',
                        help="Learning rate scheduler type.")
    parser.add_argument("--optim", type=str, default='adamw_torch_fused',
                        help="Optimizer to use.")

    args = parser.parse_args()

    # 2. 변수 설정
    # var_AUTH_TOKEN은 더 이상 필요하지 않습니다. Hugging Face 캐시를 사용합니다.
    var_TRAIN_OUTPUT_DIR = args.adapter_output_dir
    var_MERGE_OUTPUT_DIR = args.merged_output_dir

    # 출력 디렉토리 생성
    os.makedirs(var_TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(var_MERGE_OUTPUT_DIR, exist_ok=True)

    # 3. BitsAndBytesConfig 설정 (인자 사용)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=eval(f"torch.{args.bnb_4bit_compute_dtype}") # 문자열을 torch.bfloat16 등으로 변환
    )

    # 4. 모델 및 토크나이저 로드 (인자 사용)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation=args.attn_implementation,
        torch_dtype=eval(f"torch.{args.bnb_4bit_compute_dtype}"),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, add_eos_token=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    model, tokenizer = setup_chat_format(model, tokenizer)

    # 5. PEFT (LoRA) 설정 (인자 사용)
    # lora_target_modules가 콤마로 구분된 문자열일 경우 리스트로 변환
    target_modules_list = args.lora_target_modules.split(',') if ',' in args.lora_target_modules else args.lora_target_modules
    peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            target_modules=target_modules_list,
            task_type="CAUSAL_LM",
    )

    # 6. TrainingArguments 설정 (인자 사용)
    args_training = TrainingArguments(
        output_dir= var_TRAIN_OUTPUT_DIR,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True if args.bnb_4bit_compute_dtype == 'bfloat16' else False, # bfloat16 사용 여부도 dtype에 따라 설정
        tf32=False, # TF32는 특정 GPU 아키텍처에서만 지원되므로 기본적으로 False
        optim=args.optim,

        num_train_epochs=args.num_train_epochs,
        max_steps=-1,
        lr_scheduler_type=args.lr_scheduler_type,

        evaluation_strategy="epoch",
        eval_steps=None,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # 7. 데이터 로딩 로직 (인자 사용)
    if args.data_file_path.endswith('.xlsx'):
        df = pd.read_excel(args.data_file_path)
        df = df.fillna('')
        # 'id' 컬럼이 있으면 제거 (학습 데이터에는 필요 없음)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        json_data = df.to_dict(orient='records')
        raw_ds = Dataset.from_dict({"text": [json.dumps(d) for d in json_data]})
        def parse_json_example(example): # 함수 이름 충돌 방지
            return json.loads(example["text"])
        raw_ds = raw_ds.map(parse_json_example)
    elif args.data_file_path.endswith('.jsonl'):
        raw_ds = load_dataset("json", data_files=args.data_file_path, split="train")
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. .xlsx 또는 .jsonl 파일을 사용해주세요.")

    # generate_messages 함수에 system_message_template 인자 전달
    # map 함수에 lambda를 사용하여 system_message_template을 전달
    processed_ds = raw_ds.map(lambda example: generate_messages(example, args.system_message), remove_columns=raw_ds.column_names)


    if len(processed_ds) < 2:
        raise ValueError("데이터셋 샘플 수가 너무 적습니다. 최소 2개 이상의 데이터가 필요합니다.")

    dataset = processed_ds.train_test_split(test_size=0.2)
    train_data = dataset["train"]
    eval_data  = dataset["test"]

    print(f"Train size: {len(train_data)}, Test size: {len(eval_data)}")

    # 8. SFTTrainer 설정 및 학습 시작
    trainer = SFTTrainer (
        model=model,
        args=args_training,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        peft_config=peft_config,
        max_seq_length=3072,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    torch.cuda.empty_cache()

    trainer.train()

    # 9. 모델 저장 및 병합
    trainer.save_model(var_TRAIN_OUTPUT_DIR)

    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        var_TRAIN_OUTPUT_DIR,
        torch_dtype=eval(f"torch.{args.bnb_4bit_compute_dtype}"), # compute_dtype에 따라 로드 dtype 설정
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(var_MERGE_OUTPUT_DIR, safe_serialization=True, max_shard_size="2GB")

    tokenizer.save_pretrained(var_MERGE_OUTPUT_DIR)

    # 10. 메모리 해제
    del model
    del peft_model
    del trainer
    del merged_model
    del tokenizer
    torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
