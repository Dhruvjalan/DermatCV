import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import Login from './components/Login';
import Signup from './components/Signup';
import UserDashboard from './components/UserDashboard';
import ChatInterface from './components/ChatInterface';
import { getCurrentUser, logout } from './components/api';
import './index.css';
import Upload from './components/Upload';

// Protected Route Component
const ProtectedRoute = ({ children, isAuthenticated }) => {
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

// Main App Layout with Navigation
const AppLayout = ({ children, userId, onLogout }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    onLogout();
    navigate('/login');
  };

  return (
    <div className="app-layout">
      <nav className="app-nav">
        <div className="nav-brand">
          <span className="nav-logo">🧘‍♀️</span>
          <span className="nav-title">Wellness Analytics</span>
        </div>
        <div className="nav-links">
          <button onClick={() => navigate('/dashboard')} className="nav-link">
            Dashboard
          </button>
          <button onClick={() => navigate('/chat')} className="nav-link">
            AI Assistant
          </button>
          <button onClick={() => navigate('/upload')} className="nav-link">
            Wellness Scan
          </button>
          <button onClick={handleLogout} className="nav-link logout-btn">
            Logout
          </button>
        </div>
      </nav>
      <main className="app-main">
        {children}
      </main>
    </div>
  );
};

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUserId, setCurrentUserId] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const user = await getCurrentUser();
      if (user) {
        setIsAuthenticated(true);
        setCurrentUserId(user.user_id);
      }
    } catch (error) {
      console.error('Auth check failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = (user) => {
    setIsAuthenticated(true);
    setCurrentUserId(user.user_id);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setCurrentUserId(null);
  };

  if (loading) {
    return (
      <div className="app-loading">
        <div className="spinner"></div>
        <p>Loading Wellness Platform...</p>
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route path="/login" element={
          isAuthenticated ? 
          <Navigate to="/dashboard" replace /> : 
          <Login onLogin={handleLogin} />
        } />
        
        <Route path="/signup" element={
          isAuthenticated ? 
          <Navigate to="/dashboard" replace /> : 
          <Signup onLogin={handleLogin} />
        } />
        
        <Route path="/dashboard" element={
          <ProtectedRoute isAuthenticated={isAuthenticated}>
            <AppLayout userId={currentUserId} onLogout={handleLogout}>
              <UserDashboard userId={currentUserId} />
            </AppLayout>
          </ProtectedRoute>
        } />
        
        <Route path="/chat" element={
          <ProtectedRoute isAuthenticated={isAuthenticated}>
            <AppLayout userId={currentUserId} onLogout={handleLogout}>
              <ChatInterface userId={currentUserId} />
            </AppLayout>
          </ProtectedRoute>
        } />
        
        <Route path="/upload" element={
          <ProtectedRoute isAuthenticated={isAuthenticated}>
            <AppLayout userId={currentUserId} onLogout={handleLogout}>
              <Upload currentUserId={currentUserId} />
            </AppLayout>
          </ProtectedRoute>
        } />
        
        <Route path="/" element={
          <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />
        } />
      </Routes>
    </Router>
  );
}

export default App;