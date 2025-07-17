# models_ml/services/data_manager.py
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import shutil
import json
from typing import List, Dict, Any # List, Dict, Any 임포트 추가

# 파일 경로 (main.py와 동일하게 유지)
UPLOAD_FOLDER = "models_ml/data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# FILE_PATH는 이제 고정된 파일이 아니라 타입에 따라 동적으로 생성됩니다.
# FILE_PATH = os.path.join(UPLOAD_FOLDER, "uploaded_data.xlsx")

# Pydantic 모델 재정의 (main.py의 해당 모델 정의와 동일해야 함)
class DataEntry(BaseModel):
    id: int
    question: str
    answer: str
    schema: str # Text-to-SQL 데이터용
    
class NewDataEntry(BaseModel):
    question: str
    answer: str
    schema: str # Text-to-SQL 데이터용

class NewOAQnAEntry(BaseModel): # OA Q&A용 Pydantic 모델 (schema 없음)
    question: str
    answer: str

class DeleteRequest(BaseModel):
    id: int

# 파일 경로를 타입에 따라 동적으로 생성하는 헬퍼 함수
def get_typed_file_path(file_type: str) -> str:
    # 예: uploaded_text-to-sql_data.xlsx 또는 uploaded_oa-qna_data.xlsx
    file_name = f"uploaded_{file_type}_data.xlsx"
    return os.path.join(UPLOAD_FOLDER, file_name)

# 헬퍼 함수: 타입에 따라 엑셀 파일 읽기
def read_excel_file_by_type(file_type: str):
    file_path_by_type = get_typed_file_path(file_type)
    if not os.path.exists(file_path_by_type):
        # 파일이 없을 경우 빈 DataFrame 반환 (헤더는 공통 헤더로)
        # OA Q&A의 경우 'schema' 컬럼이 없을 수 있으므로, 기본 컬럼 정의를 명확히 합니다.
        if file_type == "text-to-sql":
            return pd.DataFrame(columns=['id', 'question', 'answer', 'schema'])
        elif file_type == "oa-qna":
            return pd.DataFrame(columns=['id', 'question', 'answer'])
        else:
            return pd.DataFrame(columns=['id', 'question', 'answer', 'schema']) # 기본값
    try:
        df = pd.read_excel(file_path_by_type)
        df = df.fillna('')
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1, len(df) + 1))
        df['id'] = df['id'].astype(int)
        return df
    except Exception as e:
        print(f"Error reading Excel file for type {file_type}: {e}")
        if file_type == "text-to-sql":
            return pd.DataFrame(columns=['id', 'question', 'answer', 'schema'])
        elif file_type == "oa-qna":
            return pd.DataFrame(columns=['id', 'question', 'answer'])
        else:
            return pd.DataFrame(columns=['id', 'question', 'answer', 'schema'])


# 헬퍼 함수: 타입에 따라 엑셀 파일 쓰기
def write_excel_file_by_type(df: pd.DataFrame, file_type: str):
    try:
        file_path_by_type = get_typed_file_path(file_type)
        df.to_excel(file_path_by_type, index=False)
        return True
    except Exception as e:
        print(f"Error writing to Excel file for type {file_type}: {e}")
        return False

# 1. 파일 업로드 로직 (file_type 인자 추가)
async def upload_typed_data(file: UploadFile, file_type: str):
    try:
        if not file.filename.endswith('.xlsx'):
            raise HTTPException(status_code=400, detail="유효한 .xlsx 파일만 업로드할 수 있습니다.")

        typed_file_path = get_typed_file_path(file_type)
        with open(typed_file_path, "wb") as buffer:
            buffer.write(await file.read())
        return {"status": "success", "message": f"파일 '{file.filename}'이(가) 성공적으로 업로드되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류 발생: {e}")

# 2. 데이터 목록 조회 로직 (file_type 인자 추가)
def get_data_entries(file_type: str) -> Dict[str, Any]:
    df = read_excel_file_by_type(file_type)
    return {"status": "success", "data": df.to_dict(orient='records')}

# 3. 새 데이터 추가 로직 (file_type과 NewDataEntry/NewOAQnAEntry에 따라 분기)
def add_data(entry: BaseModel, file_type: str): # BaseModel을 받아 유연하게
    df = read_excel_file_by_type(file_type)
    new_id = df['id'].max() + 1 if not df.empty and 'id' in df.columns else 1
    
    new_row_data = {'id': new_id, **entry.model_dump()}
    new_row = pd.DataFrame([new_row_data])
    
    updated_df = pd.concat([df, new_row], ignore_index=True)

    if write_excel_file_by_type(updated_df, file_type):
        return {"status": "success", "message": "새로운 데이터가 성공적으로 추가되었습니다."}
    else:
        raise HTTPException(status_code=500, detail="파일 저장 중 오류가 발생했습니다.")

# 4. 데이터 업데이트 로직 (file_type 추가)
def update_data(entry: BaseModel, file_type: str): # BaseModel을 받아 유연하게
    df = read_excel_file_by_type(file_type)

    if not df.empty and entry.id in df['id'].values:
        # Pydantic 모델의 필드에 따라 동적으로 업데이트
        for field in entry.model_fields: # model_fields는 Pydantic v2에서 사용
            if field != 'id': # id 필드는 업데이트하지 않음
                df.loc[df['id'] == entry.id, field] = getattr(entry, field)

        if write_excel_file_by_type(df, file_type):
            return {"status": "success", "message": "데이터가 성공적으로 업데이트되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="파일 저장 중 오류가 발생했습니다.")
    else:
        raise HTTPException(status_code=404, detail=f"업데이트할 데이터를 찾을 수 없습니다. (ID: {entry.id})")

# 5. 데이터 삭제 로직 (file_type 추가)
def delete_data(request: DeleteRequest, file_type: str):
    df = read_excel_file_by_type(file_type)

    if not df.empty and request.id in df['id'].values:
        df = df[df['id'] != request.id]

        if write_excel_file_by_type(df, file_type):
            return {"status": "success", "message": "데이터가 성공적으로 삭제되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="파일 저장 중 오류가 발생했습니다.")
    else:
        raise HTTPException(status_code=404, detail=f"삭제할 데이터를 찾을 수 없습니다. (ID: {request.id})")