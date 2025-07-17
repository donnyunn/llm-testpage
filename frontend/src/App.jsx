// src/App.jsx
import React, { useState } from 'react'
import './App.css';

import HuggingFaceLogin from './components/HuggingFaceLogin';
import TextToSQLParamsView from './components/TextToSQLParams';
import OAQnAParamsView from './components/OAQnAParams';
import InferenceTestView from './components/InferenceTest';
import ModelManagementView from './components/ModelManagementView'; // ModelManagementView 임포트

function App() {
  // selectedLearningType으로 상태 변수명 유지 (다시 학습 유형 선택으로 사용)
  const [selectedLearningType, setSelectedLearningType] = useState('text-to-sql');

  const FASTAPI_BASE_URL = 'http://192.168.0.43:8000'; // 여러분의 Ubuntu PC IP 주소로 변경했는지 다시 확인

  const [modelPathForInferenceTest, setModelPathForInferenceTest] = useState('');

  // 드롭다운 변경 핸들러 (학습 유형 선택)
  const handleDropdownChange = (event) => { // 함수 이름도 handleDropdownChange로 원래대로
    setSelectedLearningType(event.target.value);
  };

  return (
    <div className="App">

      <HuggingFaceLogin fastapiBaseUrl={FASTAPI_BASE_URL} />
    
      <header className="App-header">
          <h2>AI모델 관리 시스템</h2>
      </header>

      <div className="main-content">
          {/* --- 상단 학습 유형 선택 드롭다운 메뉴 (원래대로) --- */}
          <div className="top-menu-selection">
              <label htmlFor="learning-type-select">학습 유형 선택: </label>
              <select
                  id="learning-type-select"
                  value={selectedLearningType}
                  onChange={handleDropdownChange}
                  style={{ padding: '8px', fontSize: '1em', marginBottom: '20px' }}
              >
                  <option value="text-to-sql">Text-to-SQL</option>
                  <option value="oa-qna">OA Q&A</option>
              </select>
          </div>

          <hr className="divider" />

          {/* --- 하단 뷰 렌더링 영역 (선택된 학습 유형에 따른 파라미터 설정) --- */}
          <div className="learning-parameters-container">
              {/* selectedLearningType 값에 따라 TextToSQLParamsView 또는 OAQnAParamsView 렌더링 */}
              {selectedLearningType === 'text-to-sql' && <TextToSQLParamsView fastapiBaseUrl={FASTAPI_BASE_URL} />}
              {selectedLearningType === 'oa-qna' && <OAQnAParamsView fastapiBaseUrl={FASTAPI_BASE_URL} />}
          </div>

          <hr className="divider" style={{ marginTop: '40px' }} />

          {/* ★★★ ModelManagementView는 이제 조건부 없이 항상 렌더링 됩니다. ★★★ */}
          {/* ★★★ 학습 유형별 설정 섹션 아래, 추론 테스트 섹션 위에 배치됩니다. ★★★ */}
          <div className="model-management-section">
              <ModelManagementView
                fastapiBaseUrl={FASTAPI_BASE_URL}
                setModelPathForInferenceTest={setModelPathForInferenceTest} // ★★★ 이 줄 추가 ★★★
              />
          </div>

      </div> {/* .main-content 닫는 태그 */}

      {/* 추론 테스트 섹션은 맨 아래에 배치 (항상 표시) */}
      <hr className="divider" style={{ marginTop: '40px' }} />
      <InferenceTestView
        fastapiBaseUrl={FASTAPI_BASE_URL}
        modelPathFromManagement={modelPathForInferenceTest} // ★★★ 이 줄 추가 ★★★
      />
    </div>
  );
}

export default App;