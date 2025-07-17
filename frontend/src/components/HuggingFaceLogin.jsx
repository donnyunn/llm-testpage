// src/components/HuggingFaceLogin.jsx
import React, { useState } from 'react';

const HuggingFaceLogin = ({ fastapiBaseUrl }) => {
  const [hfToken, setHfToken] = useState('');
  const [loginStatus, setLoginStatus] = useState('');
  const [isLoginLoading, setIsLoginLoading] = useState(false);
  const [isLoginError, setIsLoginError] = useState(false);

  const handleLogin = async () => {
    setIsLoginLoading(true);
    setLoginStatus('Hugging Face 로그인 시도 중...');
    setIsLoginError(false);

    try {
      const response = await fetch(`${fastapiBaseUrl}/huggingface/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ hf_token: hfToken }),
      });

      // 서버 응답이 OK가 아니거나 JSON 파싱 중 오류가 발생할 수 있으므로,
      // response.json() 호출 전에 response 객체가 유효한지 확인합니다.
      if (!response) {
          throw new Error("서버로부터 유효한 응답 객체를 받지 못했습니다.");
      }

      let data;
      try {
          // response.json()이 실패할 경우를 대비하여 이 호출 자체를 try-catch로 감쌉니다.
          data = await response.json();
      } catch (jsonError) {
          // 응답이 JSON 형식이 아니거나 파싱할 수 없을 때
          console.error("서버 응답이 JSON 형식이 아닙니다:", jsonError, "Raw response status:", response.status);
          // 원본 응답 텍스트를 확인하여 오류 원인 파악에 도움
          const rawText = await response.text();
          console.error("서버 응답 원본 텍스트:", rawText);
          setLoginStatus(`Hugging Face 로그인 실패: 서버 응답 파싱 오류. ${rawText.substring(0, 100)}...`); // 원본 응답 일부 표시
          setIsLoginError(true);
          setIsLoginLoading(false);
          return; // 여기서 함수 종료
      }

      // data가 null, undefined, 또는 객체가 아닌 경우 처리 (방어 로직)
      if (!data || typeof data !== 'object' || data.status === undefined) {
          console.error("서버 응답 데이터 형식이 예상과 다릅니다 (data:", data, ", typeof data:", typeof data, ")");
          throw new Error('서버 응답 데이터 형식이 유효하지 않거나, status 필드가 없습니다.');
      }

      if (response.ok && data.status === 'success') {
        setLoginStatus('Hugging Face 로그인 성공!');
        setIsLoginError(false);
      } else {
        setLoginStatus(`Hugging Face 로그인 실패: ${data.detail || data.message || '알 수 없는 오류'}`);
        setIsLoginError(true);
      }
    } catch (error) {
      console.error("네트워크 연결 또는 요청 처리 중 오류 발생:", error);
      setLoginStatus(`네트워크 오류: ${error.message || '알 수 없는 연결 문제'}`);
      setIsLoginError(true);
    } finally {
      setIsLoginLoading(false);
    }
  };

  return (
    <div style={{ marginBottom: '30px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
      <h2>Hugging Face 로그인</h2>
      <p>모델 학습 및 다운로드를 위해 Hugging Face 토큰으로 로그인해주세요.</p>
      <label style={{ display: 'block', marginBottom: '10px' }}>
        Hugging Face Read/Write 토큰:
        <input
          type="password" // 토큰은 비밀번호 타입으로 입력하여 숨깁니다.
          value={hfToken}
          onChange={(e) => setHfToken(e.target.value)}
          placeholder="hf_..."
          style={{ width: '350px', padding: '8px', margin: '5px 0', display: 'block' }}
        />
      </label>
      <button onClick={handleLogin} disabled={isLoginLoading}
        style={{
          padding: '10px 20px',
          fontSize: '16px',
          backgroundColor: '#ff9900', // Hugging Face 색상과 유사하게
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
        }}
      >
        {isLoginLoading ? '로그인 중...' : 'Hugging Face 로그인'}
      </button>
      {loginStatus && (
        <p style={{ marginTop: '10px', color: isLoginError ? 'red' : 'blue', fontWeight: 'bold' }}>
          {loginStatus}
        </p>
      )}
    </div>
  );
};

export default HuggingFaceLogin;