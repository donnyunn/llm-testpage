// src/components/TextToSQLParams.jsx
import React, { useState, useEffect } from 'react';

const TextToSQLParamsView = ({ fastapiBaseUrl }) => {
  const [modelId, setModelId] = useState('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); // 초기값
  const [systemMessage, setSystemMessage] = useState(`You are an text to SQL query translator. Users will ask you questions and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\n{schema}`);

  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingLogs, setTrainingLogs] = useState('');
  const [isError, setIsError] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');

  const [dataEntries, setDataEntries] = useState([]);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [editingIndex, setEditingIndex] = useState(null);

  const tableColumns = ['question', 'answer', 'schema'];
  const FILE_TYPE = "text-to-sql"; // ★★★ 이 컴포넌트의 파일 유형 정의 ★★★
  
  // 학습 파라미터 상태 변수들
  // 1. 성능 최적화
  const [loadIn4bit, setLoadIn4bit] = useState(true);
  const [bnb4bitComputeDtype, setBnb4bitComputeDtype] = useState('bfloat16');
  const [attnImplementation, setAttnImplementation] = useState('eager');
  
  // 2. PEFT (LoRA) 설정
  const [loraAlpha, setLoraAlpha] = useState(128);
  const [loraDropout, setLoraDropout] = useState(0.05);
  const [loraR, setLoraR] = useState(64);
  const [loraTargetModules, setLoraTargetModules] = useState('all-linear');
  
  // 3. 학습 인자
  const [trainBatchSize, setTrainBatchSize] = useState(1);
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(4);
  const [numTrainEpochs, setNumTrainEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(2e-4);
  const [lrSchedulerType, setLrSchedulerType] = useState('constant');
  const [optim, setOptim] = useState('adamw_torch_fused');

  const handleModelIdChange = (event) => {
    setModelId(event.target.value);
  }

  const fetchDataEntries = async () => {
    setIsLoadingData(true);
    try {
      const response = await fetch(`${fastapiBaseUrl}/data-entries?file_type=${FILE_TYPE}`);      
      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setDataEntries(data.data);
      } else {
        console.error('Failed to fetch data:', data.message);
        setDataEntries([]);
      }
    } catch (error) {
      console.error('Network error fetching data:', error);
      setDataEntries([]);
    } finally {
      setIsLoadingData(false);
    }
  };

  useEffect(() => {
    fetchDataEntries();
  }, [FILE_TYPE]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.xlsx')) {
      setSelectedFile(file);
      setUploadStatus(`선택된 파일: ${file.name}`);
    } else {
      setSelectedFile(null);
      setUploadStatus('유효한 .xlsx 파일을 선택해주세요.');
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('먼저 파일을 선택해주세요.');
      return;
    }

    setUploadStatus('파일 업로드 중...');
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${fastapiBaseUrl}/upload-text-to-sql-data`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus(`파일 업로드 성공: ${data.message}`);
        await fetchDataEntries();
      } else {
        setUploadStatus(`파일 업로드 실패: ${data.detail || '알 수 없는 오류'}`);
      }
    } catch (error) {
      setUploadStatus(`네트워크 오류: ${error.message}`);
    }
  };

  const handleStartTraining = async () => {
    setTrainingStatus('학습 시작 중...');
    setTrainingLogs('서버로부터 응답 대기 중...');
    setIsError(false);
    console.log('학습 시작 버튼 클릭');
    console.log('선택된 모델 ID:', modelId);

    const trainingParams = {
        model_id: modelId,
        system_message: systemMessage,
        load_in_4bit: loadIn4bit,
        bnb_4bit_compute_dtype: bnb4bitComputeDtype,
        attn_implementation: attnImplementation,
        lora_alpha: loraAlpha,
        lora_dropout: loraDropout,
        lora_r: loraR,
        lora_target_modules: loraTargetModules,
        per_device_train_batch_size: trainBatchSize,
        gradient_accumulation_steps: gradientAccumulationSteps,
        num_train_epochs: numTrainEpochs,
        learning_rate: learningRate,
        lr_scheduler_type: lrSchedulerType,
        optim: optim,
        file_type: FILE_TYPE,
    };
    console.log("전송할 학습 파라미터:", trainingParams);

    try {
      const response = await fetch(`${fastapiBaseUrl}/start_training_test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingParams),
      });

      const data = await response.json();

      if (response.ok) {
        setTrainingStatus(`학습 요청 성공: ${data.message}`);
        setTrainingLogs(data.logs || '로그 내용 없음');
        setIsError(false);
        console.log('서버 응답:', data);
      } else {
        setTrainingStatus(`학습 요청 실패: ${data.detail || '알 수 없는 오류'}`);
        setTrainingLogs(data.logs || '로그 내용 없음');
        setIsError(true);
        console.error('서버 오류:', data);
      }
    } catch (error) {
      setTrainingStatus(`네트워크 오류 발생: ${error.message}`);
      setTrainingLogs(`오류: ${error.message}`);
      setIsError(true);
      console.error('Fetch 에러:', error);
    }
  };

  const handleEditChange = (index, field, value) => {
    const newDataEntries = [...dataEntries];
    newDataEntries[index][field] = value;
    setDataEntries(newDataEntries);
  };

  const handleSave = async (index) => {
    const entryToSave = dataEntries[index];
    console.log('데이터 저장 요청:', entryToSave);

    const isNew = entryToSave.id === null;
    const endpoint = isNew ? `${fastapiBaseUrl}/add-data/${FILE_TYPE}` : `${fastapiBaseUrl}/update-data/${FILE_TYPE}`;    
    const requestBody = isNew ? { question: entryToSave.question, answer: entryToSave.answer, schema: entryToSave.schema } : entryToSave;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        console.log('데이터 저장 성공');
        setEditingIndex(null);
        await fetchDataEntries();
      } else {
        const errorData = await response.json();
        console.error('데이터 저장 실패:', errorData);
      }
    } catch (error) {
      console.error('네트워크 오류:', error);
    }
  };

  const handleDelete = async (index) => {
    const entryToDelete = dataEntries[index];
    console.log('데이터 삭제 요청:', entryToDelete);

    try {
      const response = await fetch(`${fastapiBaseUrl}/delete-data/${FILE_TYPE}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ id: entryToDelete.id }),
      });

      if (response.ok) {
        console.log('데이터 삭제 성공');
        await fetchDataEntries(); // 삭제 후 데이터 새로고침
      } else {
        const errorData = await response.json();
        console.error('데이터 삭제 실패:', errorData);
      }
    } catch (error) {
      console.error('네트워크 오류:', error);
    }
  };

  const handleAddEntry = () => {
    setDataEntries([...dataEntries, { id: null, question: '', answer: '', schema: '' }]);
    setEditingIndex(dataEntries.length);
  }

  return (
    <div className="learning-params-section">
      <h2>Text-to-SQL 학습 설정</h2>

      {/* --- 1. 핵심 설정 섹션 --- */}
      <div className="setting-group">
        <h3>1. 핵심 설정</h3>
        <div className="form-item">
          <h4>모델 선택</h4>
          <label htmlFor="model-id">어떤 모델로 학습할지 선택 (Model ID):</label>
          <input
            type="text"
            id="model-id"
            value={modelId}
            onChange={handleModelIdChange}
            placeholder="예: google/gemma-7b-it"
            style={{ width: '300px', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
          />
        </div>
      </div>
      
      {/* 2. 시스템 메시지 입력 UI */}
      <div style={{ marginTop: '20px', marginBottom: '20px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        <h3>2. 시스템 메시지 설정</h3>
        <p>AI 모델의 역할을 정의하는 메시지를 입력하세요. `{'{schema}'}`는 자동으로 대체됩니다.</p>
        <textarea
          value={systemMessage}
          onChange={(e) => setSystemMessage(e.target.value)}
          rows="5"
          style={{ width: '100%', padding: '8px', boxSizing: 'border-box' }}
        />
      </div>

      {/* 3. 학습 데이터 업로드 섹션 */}
      <h3>3. 학습 데이터 업로드</h3>
      <p>학습에 사용할 .xlsx 파일을 업로드하세요. (헤더: id, question, answer, schema)</p>
      <input
        type="file"
        accept=".xlsx"
        onChange={handleFileChange}
        style={{ display: 'block', marginBottom: '10px' }}
      />
      <button
        onClick={handleFileUpload}
        style={{
          padding: '8px 15px',
          backgroundColor: '#28a745',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
        }}
      >
        파일 업로드
      </button>
      {uploadStatus && (
        <p style={{ marginTop: '10px', color: uploadStatus.includes('실패') ? 'red' : 'blue', fontWeight: 'bold' }}>
          {uploadStatus}
        </p>
      )}

      {/* 3. 데이터 테이블 섹션 추가 */}
      <h3>3. 학습 데이터 목록</h3>
      <button onClick={handleAddEntry} style={{ marginBottom: '15px', background: 'lightgray'}}>
        새 데이터 추가
      </button>
      {isLoadingData ? (
        <p>데이터를 불러오는 중...</p>
      ) : dataEntries.length > 0 ? (
        <div style={{ maxHeight: '1000px', overflowY: 'auto', overflowX: 'auto', border: '1px solid #ccc', borderRadius: '5px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ backgroundColor: '#f2f2f2' }}>
              <tr>
                {tableColumns.map(key => (
                  <th key={key} style={{ padding: '8px', border: '1px solid #ddd' }}>{key}</th>
                ))}
                <th style={{ padding: '8px', border: '1px solid #ddd' }}>액션</th>
              </tr>
            </thead>
            <tbody>
              {dataEntries.map((entry, index) => (
                <tr key={typeof entry.id === 'number' ? entry.id : `new-${index}`}>
                  {tableColumns.map((key) => (
                    <td key={key} style={{ padding: '8px', border: '1px solid #ddd', verticalAlign: 'top', whiteSpace: 'pre-wrap' }}>
                      {editingIndex === index ? (
                        <textarea
                          value={entry[key] || ''}
                          onChange={(e) => handleEditChange(index, key, e.target.value)}
                          style={{ width: '100%', minHeight: '50px', boxSizing: 'border-box' }}
                        />
                      ) : (
                        entry[key]
                      )}
                    </td>
                  ))}
                  <td style={{ padding: '8px', border: '1px solid #ddd' }}>
                    {editingIndex === index ? (
                      <button onClick={() => handleSave(index)}>저장</button>
                    ) : (
                      <>
                        <button onClick={() => setEditingIndex(index)} style={{marginLeft: '5px', backgroundColor: 'lightgray'}}>편집</button>
                        <button onClick={() => handleDelete(index)} style={{ marginLeft: '5px', backgroundColor: 'pink' }}>삭제</button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>업로드된 학습 데이터가 없습니다.</p>
      )}

      {/* --- 4. 성능 최적화 설정 섹션 --- */}
      <h3>4. 성능 최적화 설정</h3>
      <div style={{display: 'flex', gap: '20px', flexWrap: 'wrap', flexDirection: "column"}}>
        <div className="form-item">
          <label htmlFor="loadIn4bit">4비트 양자화 사용:</label>
          <input
            type="checkbox"
            id="loadIn4bit"
            checked={loadIn4bit}
            onChange={(e) => setLoadIn4bit(e.target.checked)}
          />
        </div>
        <div className="form-item">
          <label htmlFor="bnb4bitComputeDtype">연산 정밀도 (dtype):</label>
          <select
            id="bnb4bitComputeDtype"
            value={bnb4bitComputeDtype}
            onChange={(e) => setBnb4bitComputeDtype(e.target.value)}
          >
            <option value="bfloat16">bfloat16</option>
            <option value="float16">float16</option>
            <option value="float32">float32</option>
          </select>
        </div>
        <div className="form-item">
          <label htmlFor="attnImplementation">어텐션 구현:</label>
          <select
            id="attnImplementation"
            value={attnImplementation}
            onChange={(e) => setAttnImplementation(e.target.value)}
          >
            <option value="eager">eager</option>
            <option value="flash_attention_2">flash_attention_2</option>
          </select>
        </div>
      </div>

      {/* --- 6. PEFT (LoRA) 설정 섹션 --- */}
      <h3>6. PEFT (LoRA) 설정</h3>
      <div style={{display: 'flex', gap: '20px', flexWrap: 'wrap', flexDirection: "column"}}>
        <div className="form-item">
          <label htmlFor="loraR">LoRA R 값:</label>
          <input
            type="number"
            id="loraR"
            value={loraR}
            onChange={(e) => setLoraR(parseInt(e.target.value) || 1)}
            min="1"
          />
        </div>
        <div className="form-item">
          <label htmlFor="loraAlpha">LoRA Alpha:</label>
          <input
            type="number"
            id="loraAlpha"
            value={loraAlpha}
            onChange={(e) => setLoraAlpha(parseInt(e.target.value) || 1)}
            min="1"
          />
        </div>
        <div className="form-item">
          <label htmlFor="loraDropout">LoRA Dropout:</label>
          <input
            type="number"
            id="loraDropout"
            step="0.01"
            value={loraDropout}
            onChange={(e) => setLoraDropout(parseFloat(e.target.value) || 0)}
            min="0" max="1"
          />
          </div>
          <div className="form-item">
            <label htmlFor="loraTargetModules">Target Modules:</label>
            <input
              type="text"
              id="loraTargetModules"
              value={loraTargetModules}
              onChange={(e) => setLoraTargetModules(e.target.value)}
              placeholder="예: all-linear"
            />
          </div>
        </div>

      {/* --- 6. 학습 인자 섹션 --- */}
      <h3>6. 학습 인자</h3>
      <div style={{display: 'flex', gap: '20px', flexWrap: 'wrap', flexDirection: "column"}}>
        <div className="form-item">
          <label htmlFor="trainBatchSize">배치 크기:</label>
          <input
            type="number"
            id="trainBatchSize"
            value={trainBatchSize}
            onChange={(e) => setTrainBatchSize(parseInt(e.target.value) || 1)}
            min="1"
          />
        </div>
        <div className="form-item">
          <label htmlFor="gradientAccumulationSteps">그래디언트 누적 스텝:</label>
          <input
            type="number"
            id="gradientAccumulationSteps"
            value={gradientAccumulationSteps}
            onChange={(e) => setGradientAccumulationSteps(parseInt(e.target.value) || 1)}
            min="1"
          />
        </div>
        <div className="form-item">
          <label htmlFor="numTrainEpochs">에포크 수:</label>
          <input
            type="number"
            id="numTrainEpochs"
            value={numTrainEpochs}
            onChange={(e) => setNumTrainEpochs(parseInt(e.target.value) || 1)}
            min="1"
          />
        </div>
        <div className="form-item">
          <label htmlFor="learningRate">학습률:</label>
          <input
            type="number"
            id="learningRate"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0)}
            min="0" step="0.00001"
          />
        </div>
        <div className="form-item">
          <label htmlFor="lrSchedulerType">스케줄러:</label>
          <select
            id="lrSchedulerType"
            value={lrSchedulerType}
            onChange={(e) => setLrSchedulerType(e.target.value)}
          >
            <option value="constant">constant</option>
            <option value="cosine">cosine</option>
            <option value="linear">linear</option>
          </select>
        </div>
      </div>

      {/* 7. 학습 시작 버튼 및 로그 섹션 */}
      <div style={{ marginBottom: '30px' }}>
        <h3>7. 학습 시작</h3>
        <button
          onClick={handleStartTraining}
          disabled={isLoading}
          style={{
            marginTop: '10px',
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
          }}
        >
          {isLoading ? '학습 시작 중...' : '학습 시작'}
        </button>
        {trainingStatus && (
          <p style={{ marginTop: '15px', fontWeight: 'bold', color: isError ? 'red' : 'green' }}>
            {trainingStatus}
          </p>
        )}
      </div>

      {/* 학습 로그 표시 영역 */}
      {trainingLogs && (
        <>
          <h3>학습 로그:</h3>
          <pre style={{ backgroundColor: '#f0f0f0', padding: '15px', borderRadius: '5px', overflowX: 'auto', whiteSpace: 'pre-wrap', maxHeight: '400px', border: '1px solid #ddd' }}>
            {trainingLogs}
          </pre>
        </>
      )}
    </div>
  );
};

export default TextToSQLParamsView