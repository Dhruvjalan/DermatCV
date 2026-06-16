import React, { useState, useEffect } from 'react';
import './Upload.css';


export default function Upload({ currentUserId }) {
  const [userId, setUserId] = useState(currentUserId); // Swap out with auth state later
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://13.61.239.146:8000/api';

  useEffect(() => {
    fetchHistory();
  }, [userId]);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/history/${userId}`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data.history || []);
      }
    } catch (err) {
      console.error("Failed fetching scan history:", err);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setScanResult(null);
      setError('');
    }
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select or drop an image file first.');
      return;
    }

    setLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze/${userId}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Pipeline execution failed (${response.status})`);
      }

      const data = await response.json();
      setScanResult(data.dashboard_data);
      fetchHistory(); 
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during CV analysis.');
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = () => {
    if (!scanResult) return;
    
    const reportWindow = window.open('', '_blank');
    const { wellness_biometrics, segmentation, quality_metrics, actionable_interventions } = scanResult;

    reportWindow.document.write(`
      <html>
        <head>
          <title>DermatCV Analytics Executive Wellness Report</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 40px; color: #1e293b; }
            .header { border-bottom: 2px solid #3b82f6; padding-bottom: 20px; margin-bottom: 30px; }
            .title { font-size: 24px; font-weight: bold; margin: 0; color: #0f172a; }
            .meta { font-size: 14px; color: #64748b; margin-top: 5px; }
            .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 30px; }
            .card { border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; background: #f8fafc; }
            .card h3 { margin-top: 0; color: #3b82f6; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;}
            .metric { font-size: 28px; font-weight: bold; color: #0f172a; margin: 10px 0 5px 0;}
            .recommendations { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 4px; }
            .recommendations li { margin-bottom: 8px; }
            .disclaimer { font-size: 11px; color: #94a3b8; margin-top: 5px; text-align: center; }
            .condition-alert { color: #b91c1c; font-weight: bold; text-transform: capitalize; }
            @media print { .no-print { display: none; } }
          </style>
        </head>
        <body>
          <div class="header">
            <div class="title">DermatCV Enterprise Wellness Analytics Core</div>
            <div class="meta">Report Generated: ${new Date().toLocaleString()} | User Target: ${userId}</div>
          </div>
          
          <div class="grid">
            <div class="card">
              <h3>Composite Performance</h3>
              <div class="metric">${wellness_biometrics.composite_wellness_score} / 100</div>
              <p>Target Structural Region: <strong>${segmentation.detected_body_region}</strong></p>
              <p>Clinical Inference: <span class="condition-alert">${segmentation.detected_condition_inference}</span></p>
            </div>
            <div class="card">
              <h3>Biometric Breakdown</h3>
              <p>Stress Index: <strong>${wellness_biometrics.stress_index}%</strong></p>
              <p>Fatigue Index: <strong>${wellness_biometrics.fatigue_index}%</strong></p>
              <p>Hydration Level: <strong>${wellness_biometrics.hydration_level}%</strong></p>
            </div>
          </div>

          <div class="card" style="margin-bottom: 30px;">
            <h3>Computer Vision Pipeline Metadata</h3>
            <p>Capture Quality Context: <strong>${quality_metrics.status}</strong></p>
            <p>Laplacian Sharpness Index: ${quality_metrics.sharpness_index}</p>
            <p>LAB Space L-Channel Brightness: ${quality_metrics.brightness_index}</p>
          </div>

          <div class="recommendations">
            <h3 style="margin-top:0; color:#1e40af;">Actionable Interventions</h3>
            <ul>
              ${actionable_interventions.map(item => `<li>${item}</li>`).join('')}
            </ul>
          </div>
          
          <p class="disclaimer">⚠️ PROTOTYPE DEMO: Metrics are algorithmically simulated from latent vector space projections and do not represent a valid medical diagnostic evaluation.</p>
          <script>window.print();</script>
        </body>
      </html>
    `);
    reportWindow.document.close();
  };

  return (
    <div className="DermatCV-container">
      {/* Top Banner Header */}
      <header className="DermatCV-header">
        <div className="brand">
          <span className="pulse-indicator"></span>
          <h1>DermatCV Enterprise</h1>
        </div>
        <div className="badge">Wellness Analytics Pipeline v2.3.0</div>
      </header>

      <div className="DermatCV-grid">
        {/* Left Side: File Upload Panel */}
        <section className="panel upload-panel">
          <h2>Computer Vision Input</h2>
          <p className="subtitle">Submit high-resolution biometric frame for deep-learning analysis.</p>
          
          <form onSubmit={handleUploadSubmit}>
            <div className={`dropzone ${preview ? 'has-preview' : ''}`}>
              <input 
                type="file" 
                id="fileInput" 
                accept="image/*" 
                onChange={handleFileChange} 
              />
              <label htmlFor="fileInput" className="dropzone-label">
                {preview ? (
                  <img src={preview} alt="Biometric target source" className="source-preview" />
                ) : (
                  <div className="dropzone-prompt">
                    <span className="upload-icon">📷</span>
                    <p>Drag and drop biometric scan or <strong>Browse local drive</strong></p>
                    <span className="file-constraints">Accepts standard JPEG / PNG imagery</span>
                  </div>
                )}
              </label>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button 
              type="submit" 
              className={`submit-btn ${loading ? 'loading' : ''}`} 
              disabled={loading || !file}
            >
              {loading ? 'Executing Neural Networks...' : 'Initialize Analysis Pipeline'}
            </button>
          </form>
        </section>

        {/* Right Side: Execution Metrics Dashboard Display */}
        <section className="panel dashboard-panel">
          <h2>Pipeline Dashboard Realtime Metrics</h2>
          
          {!scanResult && !loading && (
            <div className="empty-dashboard">
              <span className="lock-icon">🔒</span>
              <p>Awaiting file stream input to ignite inference runtime engine.</p>
            </div>
          )}

          {loading && (
            <div className="dashboard-loading">
              <div className="spinner"></div>
              <p>Extracting MediaPipe face arrays...</p>
              <p className="subtext">Computing dynamic GradCAM patch maps & autoencoder latent projections.</p>
            </div>
          )}

          {scanResult && !loading && (
            <div className="dashboard-content animate-fade-in">
              <div className="results-header">
                <div className="score-badge">
                  <span className="score-val">{scanResult.wellness_biometrics.composite_wellness_score}</span>
                  <span className="score-lbl">Wellness Score</span>
                </div>
                <button onClick={downloadReport} className="download-btn">
                  📥 Download Analytics Report
                </button>
              </div>

              <div className="preview-comparison">
                <div className="img-box">
                  <span>GradCAM Attentive Hook</span>
                  <img src={`${BACKEND_URL}${scanResult.image_previews.processed_url}`} alt="GradCAM Activations Map" />
                </div>
              </div>

              <div className="metrics-group">
                <h3>Latent Space Metrics</h3>
                <div className="bar-stat">
                  <div className="stat-info"><span>Stress Index</span><strong>{scanResult.wellness_biometrics.stress_index}%</strong></div>
                  <div className="bar-container"><div className="bar color-stress" style={{width: `${scanResult.wellness_biometrics.stress_index}%`}}></div></div>
                </div>
                <div className="bar-stat">
                  <div className="stat-info"><span>Surface Fatigue</span><strong>{scanResult.wellness_biometrics.fatigue_index}%</strong></div>
                  <div className="bar-container"><div className="bar color-fatigue" style={{width: `${scanResult.wellness_biometrics.fatigue_index}%`}}></div></div>
                </div>
                <div className="bar-stat">
                  <div className="stat-info"><span>Hydration Level</span><strong>{scanResult.wellness_biometrics.hydration_level}%</strong></div>
                  <div className="bar-container"><div className="bar color-hydration" style={{width: `${scanResult.wellness_biometrics.hydration_level}%`}}></div></div>
                </div>
              </div>

              <div className="metadata-cards">
                <div className="meta-card">
                  <span className="meta-lbl">Detected Array</span>
                  <span className="meta-val">{scanResult.segmentation.detected_body_region}</span>
                </div>
                <div className="meta-card highlight-card">
                  <span className="meta-lbl">Clinical Inference</span>
                  <span className="meta-val condition-text">{scanResult.segmentation.detected_condition_inference}</span>
                </div>
                <div className="meta-card">
                  <span className="meta-lbl">Image Integrity</span>
                  <span className="meta-val highlight-green">{scanResult.quality_metrics.status}</span>
                </div>
              </div>

              <div className="interventions">
                <h3>Actionable Interventions</h3>
                <ul>
                  {scanResult.actionable_interventions.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Bottom Row: User Historical Record Pipeline Auditing Log */}
      <section className="panel history-panel">
        <h2>Historical Engine Scanning Audits (Persisted MongoDB State)</h2>
        <div className="table-responsive">
          <table className="history-table">
            <thead>
              <tr>
                <th>Scan Code Token</th>
                <th>Execution Timestamp</th>
                <th>Target Region</th>
                <th>Condition</th>
                <th>Wellness Metric</th>
                <th>Stress</th>
                <th>Fatigue</th>
                <th>Hydration</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', color: '#94a3b8' }}>No historical document records verified inside cluster dataset.</td>
                </tr>
              ) : (
                history.map((item) => (
                  <tr key={item.scan_id}>
                    <td className="mono">{item.scan_id}</td>
                    <td>{item.timestamp}</td>
                    <td><span className="region-tag">{item.body_part}</span></td>
                    <td><span className="condition-tag">{item.condition}</span></td>
                    <td className="weight-bold">{item.wellness_score}/100</td>
                    <td>{item.biometrics.stress}%</td>
                    <td>{item.biometrics.fatigue}%</td>
                    <td>{item.biometrics.hydration}%</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}