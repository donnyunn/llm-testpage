// src/components/InferenceTest.jsx
import React, { useState, useEffect } from 'react';

const InferenceTestView = ({ fastapiBaseUrl, modelPathFromManagement }) => {
  // 추론 테스트 관련 상태 변수들 (TextToSQLParams.jsx에서 이동)
  const [inferenceModelId, setInferenceModelId] = useState(modelPathFromManagement || 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'); // 추론용 모델 ID
  const [inferenceQuestion, setInferenceQuestion] = useState('직원별 총 판매액을 보여줘');
  const [inferenceSchema, setInferenceSchema] = useState(`
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
`);
  const [inferenceResult, setInferenceResult] = useState('');
  const [isInferenceLoading, setIsInferenceLoading] = useState(false);

  const [inferenceBnb4bitComputeDtype, setInferenceBnb4bitComputeDtype] = useState('bfloat16');
  
  useEffect(() => {
    if (modelPathFromManagement) {
      setInferenceModelId(modelPathFromManagement);
    }
  }, [modelPathFromManagement]); // modelPathFromManagement prop이 변경될 때마다 실행

  // 추론 실행 핸들러 함수 (TextToSQLParams.jsx에서 이동)
  const handleRunInference = async () => {
    setIsInferenceLoading(true);
    setInferenceResult('추론 결과 대기 중...');

    const inferenceParams = {
      model_id: inferenceModelId,
      question: inferenceQuestion,
      schema_info: inferenceSchema,
      bnb_4bit_compute_dtype: inferenceBnb4bitComputeDtype, 
    };
    console.log("전송할 추론 파라미터:", inferenceParams);

    try {
      const response = await fetch(`${fastapiBaseUrl}/run_inference`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inferenceParams),
      });

      const data = await response.json();

      if (response.ok) {
        setInferenceResult(data.predicted_sql || '결과 없음');
      } else {
        setInferenceResult(`오류: ${data.detail || '알 수 없는 오류'}`);
      }
    } catch (error) {
      setInferenceResult(`네트워크 오류: ${error.message}`);
    } finally {
      setIsInferenceLoading(false);
    }
  };

  return (
    <div className="inference-test-section" style={{ marginTop: '40px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
      <h2>추론 테스트</h2>
      <p>학습된 모델을 사용하여 새로운 질문에 대한 SQL 쿼리를 추론합니다.</p>
      
      <div style={{display: 'flex', gap: '20px', flexWrap: 'wrap', flexDirection: 'column'}}>
        <label htmlFor="inferenceModelId">추론용 모델 ID:</label>
        <input
          type="text"
          id="inferenceModelId"
          value={inferenceModelId}
          onChange={(e) => setInferenceModelId(e.target.value)}
          placeholder="예: TinyLlama/TinyLlama-1.1B-Chat-v1.0 또는 models_ml/outputs/merged/job-..."
          style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        <label htmlFor="inferenceBnb4bitComputeDtype">추론용 연산 정밀도 (dtype):</label>
        <select
            id="inferenceBnb4bitComputeDtype"
            value={inferenceBnb4bitComputeDtype}
            onChange={(e) => setInferenceBnb4bitComputeDtype(e.target.value)}
            style={{ width: '150px', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
        >
            <option value="bfloat16">bfloat16</option>
            <option value="float16">float16</option>
            <option value="float32">float32</option>
        </select>
        
        <label htmlFor="inferenceQuestion">질문 (자연어):</label>
        <textarea
          id="inferenceQuestion"
          rows="3"
          value={inferenceQuestion}
          onChange={(e) => setInferenceQuestion(e.target.value)}
          style={{ width: '100%', minHeight: '50px', boxSizing: 'border-box' }}
        />
        <label htmlFor="inferenceSchema">스키마 (CREATE TABLE 문):</label>
        <textarea
          id="inferenceSchema"
          rows="8"
          value={inferenceSchema}
          onChange={(e) => setInferenceSchema(e.target.value)}
          style={{ width: '100%', minHeight: '150px', boxSizing: 'border-box' }}
        />
        <button
          onClick={handleRunInference}
          disabled={isInferenceLoading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
          }}
        >
          {isInferenceLoading ? '추론 실행 중...' : '추론 실행'}
        </button>
        <h3>추론 결과:</h3>
        <pre style={{ backgroundColor: '#e0ffe0', padding: '15px', borderRadius: '5px', overflowX: 'auto', whiteSpace: 'pre-wrap', maxHeight: '200px', border: '1px solid #ddd' }}>
          {inferenceResult}
        </pre>
      </div>
    </div>
  );
};

export default InferenceTestView;