import React from 'react'

const OAQnAParamsView = () => {
  return (
    <div className="learning-params-section">
      <h2>OA Q&A 학습 설정</h2>
      <p>여기에 OA Q&A 모델 학습을 위한 상세 파라미터 입력 필드들이 들어갑니다.</p>
      <ul>
        <li>모델 선택</li>
        <li>질문/답변 데이터셋 업로드</li>
        <li>학습 전략 설정</li>
        <li>추가 최적화 옵션 등...</li>
      </ul>
    </div>
  );
};

export default OAQnAParamsView