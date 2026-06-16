import React, { useState, useEffect } from 'react';
import { getUserHistory, getUserById } from './api';
import './UserDashboard.css';

const UserDashboard = ({ userId }) => {
  const [userInfo, setUserInfo] = useState(null);
  const [history, setHistory] = useState([]);
  const [latestScan, setLatestScan] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (userId) {
      loadUserData();
    }
  }, [userId]);

  const loadUserData = async () => {
    setLoading(true);
    setError('');
    try {
      console.log(`Fetching data for user ID: ${userId}`);
      const [userData, historyData] = await Promise.all([
        getUserById(userId),
        getUserHistory(userId)
      ]);

      console.log('User Data:', userData);
      
      setUserInfo(userData);
      setHistory(historyData.history || []);
      
      if (historyData.history && historyData.history.length > 0) {
        setLatestScan(historyData.history[0]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const WellnessScoreCard = ({ score }) => {
    const getScoreColor = (score) => {
      if (score >= 70) return '#4caf50';
      if (score >= 50) return '#ff9800';
      return '#f44336';
    };

    const getScoreText = (score) => {
      if (score >= 70) return 'Excellent';
      if (score >= 50) return 'Good';
      return 'Needs Attention';
    };

    return (
      <div className="wellness-card">
        <h3>Overall Wellness Score</h3>
        <div className="score-display">
          <div className="score-circle" style={{ borderColor: getScoreColor(score) }}>
            <span className="score-number" style={{ color: 'black' }}>
              {score}
            </span>
            <span className="score-max">/100</span>
          </div>
          <div className="score-status" style={{ color: getScoreColor(score) }}>
            {getScoreText(score)}
          </div>
        </div>
      </div>
    );
  };

  const MetricBar = ({ label, value, color, unit = '' }) => (
    <div className="metric-bar">
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        <span className="metric-value">{value}{unit}</span>
      </div>
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${value}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="spinner"></div>
        <p>Loading your wellness data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <p>Error loading dashboard: {error}</p>
        <button onClick={loadUserData}>Retry</button>
      </div>
    );
  }

  return (
    <div className="user-dashboard">
      {/* Welcome Section */}
      <div className="welcome-section">
        <div className="welcome-header">
          <h1>Welcome, {userInfo?.full_name || 'User'}! 👋</h1>
          <p>Your personal wellness journey at a glance</p>
        </div>
        <div className="stats-summary">
          <div className="stat-card">
            <div className="stat-icon">📊</div>
            <div className="stat-info">
              <div className="stat-number">{history.length}</div>
              <div className="stat-label">Total Scans</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">📅</div>
            <div className="stat-info">
              <div className="stat-number">
                {history.length > 0 ? new Date(history[0].timestamp).toLocaleDateString() : 'N/A'}
              </div>
              <div className="stat-label">Last Scan</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">🎯</div>
            <div className="stat-info">
              <div className="stat-number">
                {latestScan ? (latestScan.wellness_score >= 70 ? 'On Track' : 'Improving') : 'Start'}
              </div>
              <div className="stat-label">Wellness Goal</div>
            </div>
          </div>
        </div>
      </div>

      {/* Latest Scan Results */}
      {latestScan ? (
        <>
          <div className="dashboard-section">
            <h2>Latest Wellness Scan</h2>
            <div className="latest-scan">
              <div className="scan-header">
                <span className="scan-date">
                  {new Date(latestScan.timestamp).toLocaleString()}
                </span>
                <span className="scan-id">ID: {latestScan.scan_id}</span>
              </div>
              
              <div className="scan-metrics">
                <WellnessScoreCard score={latestScan.wellness_score} />
                
                <div className="metrics-grid">
                  <MetricBar 
                    label="Stress Level" 
                    value={latestScan.biometrics.stress} 
                    color="#f44336"
                    unit="%"
                  />
                  <MetricBar 
                    label="Fatigue Level" 
                    value={latestScan.biometrics.fatigue} 
                    color="#ff9800"
                    unit="%"
                  />
                  <MetricBar 
                    label="Hydration Level" 
                    value={latestScan.biometrics.hydration} 
                    color="#2196f3"
                    unit="%"
                  />
                </div>

                <div className="quality-metrics">
                  <h3>Image Quality Analysis</h3>
                  <div className="quality-badges">
                    <div className="quality-badge">
                      <span>Sharpness</span>
                      <strong>{latestScan.quality.blur}</strong>
                    </div>
                    <div className="quality-badge">
                      <span>Brightness</span>
                      <strong>{latestScan.quality.brightness}</strong>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="dashboard-section">
            <h2>Personalized Recommendations <p style={{ color: 'red' }}>These are Dummy AI Generated</p></h2>
            <div className="recommendations-list">
              {latestScan.recommendations.map((rec, idx) => (
                <div key={idx} className="recommendation-item">
                  <div className="rec-icon">💡</div>
                  <div className="rec-text">{rec}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="empty-state">
          <div className="empty-icon">📸</div>
          <h3>No Scans Yet</h3>
          <p>Start your wellness journey by uploading your first scan through the <a href="/upload">Wellness Scan</a> interface!</p>
          <p className="empty-hint">💡 Tip: Click on the Chat tab and ask the AI assistant to help you upload an image</p>
        </div>
      )}

      {/* Scan History Timeline */}
      {history.length > 1 && (
        <div className="dashboard-section">
          <h2>Wellness Journey Timeline</h2>
          <div className="timeline">
            {history.slice(0, 5).map((scan, idx) => (
              <div key={scan.scan_id} className="timeline-item">
                <div className="timeline-marker">
                  <div className="timeline-dot"></div>
                  {idx < history.length - 1 && <div className="timeline-line"></div>}
                </div>
                <div className="timeline-content">
                  <div className="timeline-date">
                    {new Date(scan.timestamp).toLocaleDateString()}
                  </div>
                  <div className="timeline-score">
                    Wellness Score: <strong>{scan.wellness_score}</strong>
                  </div>
                  <div className="timeline-metrics">
                    <span>Stress: {scan.biometrics.stress}</span>
                    <span>Fatigue: {scan.biometrics.fatigue}</span>
                    <span>Hydration: {scan.biometrics.hydration}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default UserDashboard;