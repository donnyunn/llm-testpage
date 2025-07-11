// src/App.jsx
import React, { useState } from 'react'
import './App.css';

import HuggingFaceLogin from './components/HuggingFaceLogin';
import TextToSQLParamsView from './components/TextToSQLParams';
import OAQnAParamsView from './components/OAQnAParams';

function App() {
  const [selectedLearningType, setSelectedLearningType] = useState('text-to-sql');

  const FASTAPI_BASE_URL = 'http://192.168.0.43:8000';

  const handleDropdownChange = (event) => {
    setSelectedLearningType(event.target.value);
  };

  return (
    <div className="App">

      <HuggingFaceLogin fastapiBaseUrl={FASTAPI_BASE_URL} />
    
      <header className="App-header">
          <h2>AI모델 학습 테스트 화면</h2>
      </header>

      <div className="main-content">
          {/* --- 상단 드롭다운 메뉴 --- */}
          <div className="top-menu-selection">
              <label htmlFor="learning-type-select">학습 유형 선택: </label>
              <select
                  id="learning-type-select"
                  value={selectedLearningType} // 현재 상태 값을 드롭다운에 바인딩
                  onChange={handleDropdownChange}
              >
                  <option value="text-to-sql">Text-to-SQL</option>
                  <option value="oa-qna">OA Q&A</option>
              </select>
          </div>

          <hr className="divider" />

          {/* --- 하단 뷰 렌더링 영역 --- */}
          <div className="learning-parameters-container">
              {/* selectedLearningType 값에 따라 다른 컴포넌트를 조건부로 렌더링합니다. */}
              {selectedLearningType === 'text-to-sql' && <TextToSQLParamsView fastapiBaseUrl={FASTAPI_BASE_URL} />}
              {selectedLearningType === 'oa-qna' && <OAQnAParamsView fastapiBaseUrl={FASTAPI_BASE_URL} />}
          </div>
      </div>
    </div>
  );
}

export default App