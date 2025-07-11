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

      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setLoginStatus('Hugging Face 로그인 성공!');
        setIsLoginError(false);
      } else {
        setLoginStatus(`Hugging Face 로그인 실패: ${data.detail || data.message || '알 수 없는 오류'}`);
        setIsLoginError(true);
      }
    } catch (error) {
      setLoginStatus(`네트워크 오류: ${error.message}`);
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