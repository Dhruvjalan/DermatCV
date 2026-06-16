const API_BASE_URL = 'http://localhost:8000/api';

const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API Request Failed: ${response.statusText}`);
  }
  return response.json();
};

// Store user telemetry in local session
const setUserSession = (user) => {
  if (user) {
    localStorage.setItem('user_id', user.user_id);
    localStorage.setItem('user_name', user.full_name);
    localStorage.setItem('user_email', user.email);
  } else {
    localStorage.removeItem('user_id');
    localStorage.removeItem('user_name');
    localStorage.removeItem('user_email');
  }
};

export const getCurrentUser = () => {
  const userId = localStorage.getItem('user_id');
  if (!userId) return null;
  
  return {
    user_id: userId,
    full_name: localStorage.getItem('user_name'),
    email: localStorage.getItem('user_email')
  };
};

// ==========================================
// User Authentication Engine
// ==========================================

export const login = async (email, password) => {
  const response = await fetch(`${API_BASE_URL}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  const user = await handleResponse(response);
  setUserSession(user);
  return user;
};

export const signup = async (fullName, email, password,age,height,gender) => {
  const response = await fetch(`${API_BASE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // Payload maps specifically to Pydantic UserCreate schema
    body: JSON.stringify({ full_name: fullName, email, password,age,height,gender })
  });
  const user = await handleResponse(response);
  setUserSession(user);
  return user;
};

export const logout = () => {
  setUserSession(null);
};

// ==========================================
// User Management
// ==========================================

export const getUserById = async (userId) => {
  const response = await fetch(`${API_BASE_URL}/users/${userId}`);
  return handleResponse(response);
};

// ==========================================
// Computer Vision Analysis Pipeline
// ==========================================

export const analyzeScan = async (userId, file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/analyze/${userId}`, {
    method: 'POST',
    body: formData
  });
  return handleResponse(response);
};

// ==========================================
// Historical Audits
// ==========================================

export const getUserHistory = async (userId) => {
  const response = await fetch(`${API_BASE_URL}/history/${userId}`);
  return handleResponse(response);
};

// ==========================================
// Admin & System Telemetry
// ==========================================

export const getAdminRecords = async () => {
  const response = await fetch(`${API_BASE_URL}/admin/records`);
  return handleResponse(response);
};

export const getAdminSummary = async () => {
  const response = await fetch(`${API_BASE_URL}/admin/summary`);
  return handleResponse(response);
};

export const healthCheck = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(response);
};

// ==========================================
// Diagnostic RL Graph System
// ==========================================

export const startDiagnostic = async (userId, symptoms) => {
  const response = await fetch(`${API_BASE_URL}/diagnose/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      symptom_text: symptoms,
      user_id: userId,
      hyperparams: { max_questions: 6, confidence_threshold: 3.5 } 
    })
  });  

  return handleResponse(response);
};

// export const answerDiagnosticQuestion = async (sessionId, answer) => {
//   const response = await fetch(`${API_BASE_URL}/diagnose/answer`, {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify({ 
//       session_id: sessionId,
//       answer: answer
//     })
//   });

//   return handleResponse(response);
// };

export const answerDiagnosticQuestion = async (sessionId, answer) => {
  const response = await fetch(`${API_BASE_URL}/diagnose/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      session_id: sessionId,
      answer: answer
    })
  });

  return handleResponse(response);
};

// NEW: Fetch previous chat session
export const getDiagnosticHistory = async (userId) => {
  const response = await fetch(`${API_BASE_URL}/diagnose/history/${userId}`);
  return handleResponse(response);
};