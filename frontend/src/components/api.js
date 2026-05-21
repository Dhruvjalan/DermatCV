const API_BASE_URL = 'http://localhost:8000/api';

// Helper for handling responses
const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'API request failed');
  }
  return response.json();
};

// Store user data in localStorage
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

// User Authentication
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

export const signup = async (fullName, email, password) => {
  const response = await fetch(`${API_BASE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ full_name: fullName, email, password })
  });
  const user = await handleResponse(response);
  setUserSession(user);
  return user;
};

export const logout = () => {
  setUserSession(null);
};

// User Management
export const getUserById = async (userId) => {
  const response = await fetch(`${API_BASE_URL}/users/${userId}`);
  return handleResponse(response);
};

// Scan Analysis
export const analyzeScan = async (userId, file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/analyze/${userId}`, {
    method: 'POST',
    body: formData
  });
  return handleResponse(response);
};

// History
export const getUserHistory = async (userId) => {
  const response = await fetch(`${API_BASE_URL}/history/${userId}`);
  return handleResponse(response);
};

// Admin endpoints
export const getAdminRecords = async () => {
  const response = await fetch(`${API_BASE_URL}/admin/records`);
  return handleResponse(response);
};

export const getAdminSummary = async () => {
  const response = await fetch(`${API_BASE_URL}/admin/summary`);
  return handleResponse(response);
};

// Health check
export const healthCheck = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(response);
};