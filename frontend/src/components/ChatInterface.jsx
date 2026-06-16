import React, { useState, useEffect, useRef } from 'react';
import './ChatInterface.css';
import { startDiagnostic, answerDiagnosticQuestion, getDiagnosticHistory } from './api';

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'https://13-61-239-146.nip.io/api';

const ChatInterface = ({ userId }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [diagnosticStatus, setDiagnosticStatus] = useState('idle'); 
  
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (userId) {
      loadPreviousSession();
    }
  }, [userId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadPreviousSession = async () => {
    try {
      setIsTyping(true);
      const data = await getDiagnosticHistory(userId);
      
      if (data && data.session_id) {
        setSessionId(data.session_id);
        
        // Reconstruct the message array from the MongoDB qa_log
        let restoredMessages = [
          { id: 1, text: "Hello! I'm your Diagnostic AI Assistant. Let's continue your evaluation.", sender: 'bot', timestamp: new Date(data.started_at), type: 'text' },
          { id: 2, text: data.user_query, sender: 'user', timestamp: new Date(data.started_at), type: 'text' }
        ];

        let msgId = 3;
        data.qa_log.forEach((log) => {
          restoredMessages.push({
            id: msgId++, text: log.question, sender: 'bot', type: 'question', timestamp: new Date()
          });
          if (log.answer) {
            restoredMessages.push({
              id: msgId++, text: log.answer, sender: 'user', type: 'text', timestamp: new Date()
            });
          }
        });

        // Determine status based on termination reason
        // Determine status based on termination reason
        // Determine status based on termination reason
        if (data.termination_reason || data.final_diagnosis) {
          setDiagnosticStatus('complete');
          restoredMessages.push({
            id: msgId++, 
            sender: 'bot', 
            type: 'report',
            diagnosis: data.final_diagnosis,
            reportPath: `/static/report_${data.session_id}.txt`,
            timestamp: new Date() // <-- ADD THIS LINE
          });
        } else {
          setDiagnosticStatus('ongoing');
        }

        setMessages(restoredMessages);
      } else {
        // No history found, set default greeting
        setMessages([{
          id: 1, text: "Hello! I'm your Diagnostic AI Assistant. Please describe your symptoms in detail to begin your evaluation.", sender: 'bot', timestamp: new Date(), type: 'text'
        }]);
      }
    } catch (error) {
      console.log('No previous chat history found or new user.');
      setMessages([{
        id: 1, text: "Hello! I'm your Diagnostic AI Assistant. Please describe your symptoms in detail to begin your evaluation.", sender: 'bot', timestamp: new Date(), type: 'text'
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const addMessage = (text, sender, type = 'text', additionalData = null) => {
    setMessages(prev => [...prev, {
      id: prev.length + 1, text, sender, timestamp: new Date(), type, ...additionalData
    }]);
  };

  const handleStartDiagnosis = async (symptoms) => {
    try {
      console.log("Starting diagnosis with symptoms:", symptoms);
      const response = await startDiagnostic(userId, symptoms);
      if (response.session_id) {
        setSessionId(response.session_id);
        setDiagnosticStatus(response.status); 
        addMessage(response.question, 'bot', 'question', { target_symptom: response.target_symptom });
      }
    } catch (error) {
      addMessage("I'm having trouble connecting to the diagnostic engine. Please ensure the backend is running.", 'bot');
    } finally {
      setIsTyping(false);
    }
  };

  const handleAnswerQuestion = async (answer) => {
    try {
      const data = await answerDiagnosticQuestion(sessionId, answer);
      setDiagnosticStatus(data.status);

      if (data.status === 'ongoing') {
        addMessage(data.question, 'bot', 'question', { target_symptom: data.target_symptom });
      } else if (data.status === 'complete') {
        addMessage("Evaluation complete. Here are your results:", 'bot', 'report', {
          diagnosis: data.diagnosis,
          differentials: data.ranked_differentials,
          reportPath: data.report_download_url
        });
      }
    } catch (error) {
      addMessage("There was an error processing your response.", 'bot');
    } finally {
      setIsTyping(false);
    }
  };

  const sendMessage = async (overrideText = null) => {
    const textToSend = overrideText !== null ? overrideText : inputMessage;
    if (!textToSend.trim()) return;

    addMessage(textToSend, 'user');
    setInputMessage('');
    setIsTyping(true);

    if (diagnosticStatus === 'idle' || !sessionId) {
      await handleStartDiagnosis(textToSend);
    } else if (diagnosticStatus === 'ongoing') {
      await handleAnswerQuestion(textToSend);
    } else if (diagnosticStatus === 'complete') {
       // If complete, typing again starts a NEW session
       setSessionId(null);
       setDiagnosticStatus('idle');
       await handleStartDiagnosis(textToSend);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const downloadReport = async (filepath) => {
    const filename = filepath.split('/').pop().split('\\').pop();
    window.open(`${API_BASE_URL}/reports/download/${filename}`, '_blank');
  };

  const renderQuickActions = () => {
    if (diagnosticStatus === 'ongoing') {
      return (
        <div className="quick-questions">
          {['Yes', 'No', 'Not sure', 'Skip'].map((btn, idx) => (
            <button key={idx} className="quick-question" onClick={() => sendMessage(btn)}>
              {btn}
            </button>
          ))}
        </div>
      );
    }
    
    return (
      <div className="quick-questions">
        {["I have red itchy patches on my skin", "I have a severe headache and nausea", "My joints are swelling"].map((btn, idx) => (
          <button key={idx} className="quick-question" onClick={() => setInputMessage(btn)}>
            {btn}
          </button>
        ))}
      </div>
    );
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="chat-header-info">
          <h2>💬 AyurGenx Diagnostic Engine</h2>
          <p>Powered by Knowledge Graph RAG</p>
        </div>
        <div className="context-badge">
          {diagnosticStatus === 'ongoing' ? (
            <span className="status-ongoing">🔄 Evaluating Data...</span>
          ) : (
            <span>✨ Ready</span>
          )}
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-avatar">
              {message.sender === 'bot' ? '🤖' : '👤'}
            </div>
            <div className="message-content">
              
              {(message.type === 'text' || message.type === 'question') && (
                <div className="message-text">{message.text}</div>
              )}

              {message.type === 'report' && message.diagnosis && (
                <div className="message-report-card">
                  <h4>Primary Hypothesis</h4>
                  <p className="primary-diagnosis">{message.diagnosis.disease.toUpperCase()}</p>
                  <p className="confidence-score">Graph Match Score: {message.diagnosis.score}</p>
                  
                  {message.differentials && message.differentials.length > 1 && (
                    <div className="differentials">
                      <h5>Secondary Considerations:</h5>
                      <ul>
                        {message.differentials.slice(1).map((diff, i) => (
                          <li key={i}>{diff.disease} ({diff.score})</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {message.reportPath && (
                    <button 
                      className="download-btn"
                      onClick={() => downloadReport(message.reportPath)}
                    >
                      📄 Download Full Clinical Report
                    </button>
                  )}
                </div>
              )}

              <div className="message-time">
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="message bot">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {renderQuickActions()}

      <div className="chat-input-container">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={diagnosticStatus === 'ongoing' ? "Type your answer..." : "Describe your symptoms in detail..."}
          rows="2"
        />
        <button onClick={() => sendMessage()} disabled={!inputMessage.trim() || isTyping}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
