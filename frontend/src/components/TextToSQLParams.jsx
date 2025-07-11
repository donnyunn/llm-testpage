import React, { useState, useEffect } from 'react';

const TextToSQLParamsView = ({ fastapiBaseUrl }) => {
  const [modelId, setModelId] = useState('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); // 초기값
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
  const allColumns = ['id', ...tableColumns]; // 내부적으로 id도 관리

  const handleModelIdChange = (event) => {
    setModelId(event.target.value);
  }

  const fetchDataEntries = async () => {
    setIsLoadingData(true);
    try {
      const response = await fetch(`${fastapiBaseUrl}/data-entries`);
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
  }, []);

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
      const response = await fetch(`${fastapiBaseUrl}/upload-data`, {
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

    try {
      const response = await fetch(`${fastapiBaseUrl}/start_training_test` , {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelId,
        }),
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
    const endpoint = isNew ? `${fastapiBaseUrl}/add-data` : `${fastapiBaseUrl}/update-data`;
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
      const response = await fetch(`${fastapiBaseUrl}/delete-data`, {
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
              <label htmlFor="model-id">어떤 모델로 학습할지 선택 (Model ID):</label>
              <input
                  type="text"
                  id="model-id"
                  value={modelId} // modelId state와 입력 필드를 연결합니다.
                  onChange={handleModelIdChange} // 입력 값이 변경될 때 state를 업데이트합니다.
                  placeholder="예: google/gemma-7b-it"
                  style={{ width: '300px', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
              />
          </div>
      </div>

      {/* 2. 데이터 업로드 섹션 추가 */}
      <div style={{ marginBottom: '30px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        <h3>2. 학습 데이터 업로드</h3>
        <p>학습에 사용할 .xlsx 파일을 업로드하세요. (헤더: question, answer, schema)</p>
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
      </div>

      {/* 3. 데이터 테이블 섹션 추가 */}
      <div style={{ marginBottom: '30px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
        <h3>3. 학습 데이터 목록</h3>
        <button onClick={handleAddEntry} style={{ marginBottom: '15px' }}>
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
                          <button onClick={() => setEditingIndex(index)}>편집</button>
                          <button onClick={() => handleDelete(index)} style={{ marginLeft: '5px', backgroundColor: 'red', color: 'white' }}>삭제</button>
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
      </div>

      {/* 4. 학습 시작 버튼 및 로그 섹션 */}
      <div style={{ marginBottom: '30px' }}>
        <h3>4. 학습 시작</h3>
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