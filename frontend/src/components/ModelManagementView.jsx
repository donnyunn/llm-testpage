// src/components/ModelManagementView.jsx
import React, { useState, useEffect } from 'react';

const ModelManagementView = ({ fastapiBaseUrl, setModelPathForInferenceTest }) => {
  const [models, setModels] = useState([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [activeModel, setActiveModel] = useState(null);

  // 모델 목록을 DB에서 불러오는 함수
  const fetchModels = async () => {
    setIsLoadingModels(true);
    try {
      const response = await fetch(`${fastapiBaseUrl}/api/models`);
      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setModels(data.data);
        const active = data.data.find(model => model.status === 'deployed');
        setActiveModel(active);
        console.log("모델 목록 불러오기 성공:", data.data);
      } else {
        console.error('Failed to fetch models:', data.message || data.detail || '알 수 없는 오류');
        setModels([]);
      }
    } catch (error) {
      console.error('Network error fetching models:', error);
      setModels([]);
    } finally {
      setIsLoadingModels(false);
    }
  };

  // 컴포넌트 마운트 시 모델 목록 불러오기
  useEffect(() => {
    fetchModels();
  }, []);

  // ★★★ 배포 버튼 클릭 핸들러 (새로 추가) ★★★
  const handleDeploy = async (jobId) => {
    if (!window.confirm(`정말로 Job ID "${jobId}" 모델을 배포하시겠습니까? 현재 활성 모델은 비활성화됩니다.`)) {
      return; // 사용자가 취소한 경우
    }
    try {
      const response = await fetch(`${fastapiBaseUrl}/api/models/activate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ job_id: jobId }),
      });
      const data = await response.json();
      if (response.ok && data.status === 'success') {
        alert(data.message);
        await fetchModels(); // 성공 시 목록 새로고침
      } else {
        alert(`배포 실패: ${data.detail || data.message || '알 수 없는 오류'}`);
      }
    } catch (error) {
      alert(`네트워크 오류: ${error.message}`);
    }
  };

  // ★★★ 삭제 버튼 클릭 핸들러 (새로 추가) ★★★
  const handleDelete = async (jobId) => {
    if (!window.confirm(`정말로 Job ID "${jobId}" 모델을 삭제하시겠습니까? (물리적 파일도 삭제됩니다!)`)) {
      return; // 사용자가 취소한 경우
    }
    try {
      const response = await fetch(`${fastapiBaseUrl}/api/models/delete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ job_id: jobId }),
      });
      const data = await response.json();
      if (response.ok && data.status === 'success') {
        alert(data.message);
        await fetchModels(); // 성공 시 목록 새로고침
      } else {
        alert(`삭제 실패: ${data.detail || data.message || '알 수 없는 오류'}`);
      }
    } catch (error) {
      alert(`네트워크 오류: ${error.message}`);
    }
  };

  // ★★★ 테스트 버튼 클릭 핸들러 (새로 추가 - 임시) ★★★
  // 이 부분은 향후 InferenceTestView와 직접 연동하도록 개선됩니다.
  const handleTest = (jobId, mergedPath) => {
    // alert(`Job ID "${jobId}" 모델로 추론 테스트를 시작합니다. \n\n 추론 테스트 섹션의 "추론용 모델 ID"에 다음 경로를 직접 입력하세요: \n ${mergedPath}`);
    // TODO: InferenceTestView의 inferenceModelId를 자동으로 설정하는 로직 필요 (React Context API 등)
    // 이제 alert 대신 setModelPathForInferenceTest 함수를 호출합니다.
    if (setModelPathForInferenceTest) { // prop이 존재하는지 확인 (방어 로직)
      setModelPathForInferenceTest(mergedPath); // 추론 테스트 컴포넌트의 모델 ID를 설정
      alert(`Job ID "${jobId}" 모델 경로가 추론 테스트 섹션에 설정되었습니다. 이제 추론 테스트 섹션으로 이동하여 "추론 실행" 버튼을 누르세요.`);
      // TODO: 스크롤하여 InferenceTestView로 자동 이동하는 기능 추가 (나중에)
    } else {
      alert("추론 테스트 기능 설정 오류: setModelPathForInferenceTest 함수를 찾을 수 없습니다.");
    }
  };

  // 테이블 컬럼 정의 (표시에 사용할 키와 헤더명)
  const tableColumns = [
    { key: 'job_id', label: 'Job ID' },
    { key: 'base_model_id', label: '기반 모델' },
    { key: 'training_date', label: '학습 일자' },
    { key: 'eval_accuracy', label: '정확도' },
    { key: 'eval_loss', label: '손실' },
    { key: 'lora_r', label: 'LoRA R' },
    { key: 'status', label: '상태' },
    { key: 'description', label: '설명' },
  ];

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>모델 관리 및 배포</h2>
      <p>학습된 AI 모델들의 목록을 확인하고 관리합니다.</p>

      {/* 활성 모델 표시 */}
      {activeModel ? (
        <div style={{ border: '1px solid #007bff', padding: '15px', borderRadius: '8px', marginBottom: '20px', backgroundColor: '#e6f7ff' }}>
          <h3>현재 활성 모델: {activeModel.job_id}</h3>
          <p>기반 모델: {activeModel.base_model_id}</p>
          <p>정확도: {activeModel.eval_accuracy !== null ? activeModel.eval_accuracy.toFixed(4) : 'N/A'}</p>
          <p>학습 일자: {new Date(activeModel.training_date).toLocaleString()}</p>
        </div>
      ) : (
        <p>현재 배포된 활성 모델이 없습니다.</p>
      )}

      <button onClick={fetchModels} style={{ marginBottom: '15px', padding: '10px 15px', backgroundColor: '#6c757d', color: 'white', border: 'none', borderRadius: '5px' }}>
        목록 새로고침
      </button>

      {isLoadingModels ? (
        <p>모델 목록을 불러오는 중...</p>
      ) : models.length > 0 ? (
        <div style={{ maxHeight: '600px', overflowY: 'auto', overflowX: 'auto', border: '1px solid #ccc', borderRadius: '5px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ backgroundColor: '#f2f2f2', position: 'sticky', top: 0, zIndex: 1 }}>
              <tr>
                {tableColumns.map(col => (
                  <th key={col.key} style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>{col.label}</th>
                ))}
                <th style={{ padding: '8px', border: '1px solid #ddd', textAlign: 'left' }}>액션</th>
              </tr>
            </thead>
            <tbody>
              {models.map((modelEntry) => (
                <tr key={modelEntry.job_id}>
                  {tableColumns.map(col => (
                    <td key={col.key} style={{ padding: '8px', border: '1px solid #ddd', verticalAlign: 'top', whiteSpace: 'pre-wrap' }}>
                      {/* 날짜 형식 변환 및 숫자 소수점 처리 예시 */}
                      {col.key === 'training_date' ? new Date(modelEntry[col.key]).toLocaleString() :
                       (col.key === 'eval_accuracy' || col.key === 'eval_loss') && modelEntry[col.key] !== null ? modelEntry[col.key].toFixed(4) :
                       modelEntry[col.key]}
                    </td>
                  ))}
                  <td style={{ padding: '8px', border: '1px solid #ddd' }}>
                    {/* ★★★ onClick 핸들러 추가 ★★★ */}
                    <button onClick={() => handleDeploy(modelEntry.job_id)} style={{ backgroundColor: '#007bff', color: 'white', border: 'none', padding: '5px 10px', borderRadius: '3px' }}>배포</button>
                    <button onClick={() => handleTest(modelEntry.job_id, modelEntry.merged_path)} style={{ backgroundColor: '#ffc107', color: 'black', border: 'none', padding: '5px 10px', borderRadius: '3px', marginLeft: '5px' }}>테스트</button>
                    <button onClick={() => handleDelete(modelEntry.job_id)} style={{ backgroundColor: '#dc3545', color: 'white', border: 'none', padding: '5px 10px', borderRadius: '3px', marginLeft: '5px' }}>삭제</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>등록된 모델이 없습니다. 학습을 시작하여 모델을 생성해주세요.</p>
      )}
    </div>
  );
};

export default ModelManagementView;