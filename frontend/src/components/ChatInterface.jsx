import React, { useState, useEffect, useRef } from 'react';
import { getUserHistory } from './api';
import './ChatInterface.css';

const ChatInterface = ({ userId }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your Wellness AI Assistant. I can help you understand your scan results, provide wellness advice, and answer questions about your health metrics. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [userContext, setUserContext] = useState(null);
  const messagesEndRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load user context when userId changes
  useEffect(() => {
    if (userId) {
      loadUserContext();
    }
  }, [userId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadUserContext = async () => {
    try {
      const history = await getUserHistory(userId);
      if (history && history.history && history.history.length > 0) {
        const latestScan = history.history[0];
        setUserContext({
          hasScans: true,
          latestWellnessScore: latestScan.wellness_score,
          latestBiometrics: latestScan.biometrics,
          totalScans: history.total_scans
        });
        
        // Add contextual message
        addBotMessage(`I see you've had ${history.total_scans} wellness scan(s). Your latest wellness score was ${latestScan.wellness_score}. Would you like to discuss your results?`);
      } else {
        setUserContext({ hasScans: false });
        addBotMessage("You haven't done any wellness scans yet. I recommend starting with a scan to get personalized insights!");
      }
    } catch (error) {
      console.error('Error loading user context:', error);
    }
  };

  const addBotMessage = (text) => {
    const botMessage = {
      id: messages.length + 1,
      text: text,
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, botMessage]);
  };

  const getBotResponse = (userInput) => {
    const input = userInput.toLowerCase();
    
    // Wellness score related queries
    if (input.includes('wellness score') || input.includes('my score')) {
      if (userContext?.latestWellnessScore) {
        const score = userContext.latestWellnessScore;
        let advice = '';
        if (score >= 70) {
          advice = "Great job! Your wellness score is excellent. Keep maintaining your healthy habits!";
        } else if (score >= 50) {
          advice = "Your wellness score is moderate. There's room for improvement. Would you like specific recommendations?";
        } else {
          advice = "Your wellness score indicates room for improvement. I strongly recommend reviewing your lifestyle habits and considering the actionable interventions from your scan.";
        }
        return `Your latest wellness score is ${score}/100. ${advice}`;
      } else {
        return "You don't have any scan results yet. Please complete a wellness scan first to get your score.";
      }
    }
    
    // Stress related queries
    if (input.includes('stress') || input.includes('anxiety')) {
      if (userContext?.latestBiometrics) {
        const stress = userContext.latestBiometrics.stress;
        if (stress > 70) {
          return `Your stress index is ${stress}/100, which is quite high. I recommend practicing deep breathing exercises, taking regular breaks, and considering mindfulness meditation. Would you like some guided breathing techniques?`;
        } else if (stress > 40) {
          return `Your stress index is ${stress}/100 - moderate. Regular exercise, adequate sleep, and work-life balance can help reduce this further.`;
        } else {
          return `Great news! Your stress index is ${stress}/100, indicating good stress management. Keep up your healthy routines!`;
        }
      }
      return "I can help with stress management techniques. Would you like to learn some breathing exercises or meditation tips?";
    }
    
    // Fatigue queries
    if (input.includes('fatigue') || input.includes('tired') || input.includes('energy')) {
      if (userContext?.latestBiometrics) {
        const fatigue = userContext.latestBiometrics.fatigue;
        if (fatigue > 70) {
          return `Your fatigue index is ${fatigue}/100. This suggests you might benefit from improving your sleep quality, staying hydrated, and taking regular movement breaks throughout the day.`;
        } else if (fatigue > 40) {
          return `Your fatigue level is ${fatigue}/100 - moderate. Ensure you're getting 7-8 hours of sleep and staying active.`;
        } else {
          return `Your energy levels look good with a fatigue index of ${fatigue}/100! Keep maintaining your healthy sleep and exercise routines.`;
        }
      }
      return "Fatigue can be managed with proper sleep, nutrition, and stress management. Would you like specific advice?";
    }
    
    // Hydration queries
    if (input.includes('hydrat') || input.includes('water') || input.includes('drink')) {
      if (userContext?.latestBiometrics) {
        const hydration = userContext.latestBiometrics.hydration;
        if (hydration < 40) {
          return `Your hydration level is ${hydration}/100, which is low. I strongly recommend increasing your water intake to at least 2-3 liters per day. Set reminders to drink water regularly!`;
        } else if (hydration < 70) {
          return `Your hydration is at ${hydration}/100 - good but could be better. Try to drink a glass of water every hour during work hours.`;
        } else {
          return `Excellent! Your hydration level is ${hydration}/100. You're doing great at staying hydrated. Keep it up!`;
        }
      }
      return "Staying hydrated is crucial for wellness. Aim for 2-3 liters of water daily. Would you like tips on tracking your water intake?";
    }
    
    // Recommendations
    if (input.includes('recommend') || input.includes('advice') || input.includes('improve')) {
      return "Based on wellness best practices, I recommend:\n• Stay hydrated (2-3L water daily)\n• Get 7-8 hours of quality sleep\n• Practice daily stress management (meditation, deep breathing)\n• Take regular movement breaks\n• Maintain a balanced diet rich in antioxidants\n\nWould you like detailed guidance on any of these?";
    }
    
    // Scan history
    if (input.includes('history') || input.includes('previous') || input.includes('past scan')) {
      if (userContext?.totalScans) {
        return `You have ${userContext.totalScans} scan(s) in your history. Your most recent wellness score was ${userContext.latestWellnessScore}. Would you like to see a detailed comparison of your progress?`;
      }
      return "You don't have any scan history yet. Start with a wellness scan to track your progress over time!";
    }
    
    // General health tips
    if (input.includes('health tip') || input.includes('wellness tip')) {
      const tips = [
        "Take a 5-minute breathing break every 2 hours to reset your nervous system.",
        "Blue light from screens can affect sleep. Try using night mode after 7 PM.",
        "Cold exposure (like splashing cold water on your face) can boost alertness and reduce stress.",
        "Regular stretching improves blood flow and reduces physical fatigue.",
        "Laughter truly is medicine - it reduces stress hormones and boosts immune function."
      ];
      return tips[Math.floor(Math.random() * tips.length)];
    }
    
    // Greetings
    if (input.includes('hello') || input.includes('hi') || input.includes('hey')) {
      return "Hello! How are you feeling today? I can help analyze your wellness metrics or provide health tips.";
    }
    
    // Help
    if (input.includes('help') || input.includes('what can you do')) {
      return "I can help you with:\n• Understanding your wellness scores\n• Managing stress and fatigue\n• Hydration advice\n• Wellness recommendations\n• Scan history analysis\n• Daily health tips\n\nJust ask me about any of these topics!";
    }
    
    // Default response
    return "I'm here to support your wellness journey. You can ask me about your wellness scores, stress levels, fatigue, hydration, or general health tips. What would you like to know?";
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    // Add user message
    const userMsg = {
      id: messages.length + 1,
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsTyping(true);

    // Simulate bot thinking
    setTimeout(() => {
      const response = getBotResponse(inputMessage);
      const botMsg = {
        id: messages.length + 2,
        text: response,
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMsg]);
      setIsTyping(false);
    }, 800);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const quickQuestions = [
    "What's my wellness score?",
    "How can I reduce stress?",
    "Give me a health tip",
    "Help with fatigue",
    "Hydration advice",
    "Show my history"
  ];

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="chat-header-info">
          <h2>💬 Wellness AI Assistant</h2>
          <p>Your personal health companion</p>
        </div>
        {userContext && (
          <div className="context-badge">
            {userContext.hasScans ? (
              <span>📊 {userContext.totalScans} scans | Latest: {userContext.latestWellnessScore}/100</span>
            ) : (
              <span>✨ Ready for first scan</span>
            )}
          </div>
        )}
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-avatar">
              {message.sender === 'bot' ? '🤖' : '👤'}
            </div>
            <div className="message-content">
              <div className="message-text">{message.text}</div>
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
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="quick-questions">
        {quickQuestions.map((question, idx) => (
          <button
            key={idx}
            className="quick-question"
            onClick={() => {
              setInputMessage(question);
              setTimeout(() => sendMessage(), 100);
            }}
          >
            {question}
          </button>
        ))}
      </div>

      <div className="chat-input-container">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message here... (Press Enter to send)"
          rows="2"
        />
        <button onClick={sendMessage} disabled={!inputMessage.trim() || isTyping}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;