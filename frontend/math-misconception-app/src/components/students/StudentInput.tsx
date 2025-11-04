import React, { useState } from 'react';
import { MisconceptionAnalysis } from '@/utils/api';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import { apiClient } from '@/utils/api';
import '@/styles/components/StudentInput.css';

interface StudentInputProps {
  onAnalysisComplete: (result: MisconceptionAnalysis) => void;
  onAnalysisStart?: () => void;
  onAnalysisEnd?: () => void;
}

const StudentInput: React.FC<StudentInputProps> = ({ 
  onAnalysisComplete, 
  onAnalysisStart, 
  onAnalysisEnd 
}) => {
  const [statement, setStatement] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!statement.trim()) {
      setError('Please enter a student statement');
      return;
    }

    setIsLoading(true);
    setError(null);
    onAnalysisStart?.();

    try {
      const result = await apiClient.analyzeStatement(statement);
      onAnalysisComplete(result);
      setStatement('');
    } catch (err) {
      setError('Failed to analyze statement. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
      onAnalysisEnd?.();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAnalyze();
    }
  };

  const examples = [
    "355 is bigger than 8 so 0.355 is larger",
    "I added all sides to find area",
    "The number without x is the slope",
    "I added before multiplying because it came first",
    "1/3 is bigger than 1/2 because 3 is bigger than 2",
    "0.5 is the same as 5%"
  ];

  return (
    <div className="student-input-container">
      {/* Floating Background Elements */}
      <div className="input-floating-elements">
        <div className="floating-input-icon icon-1">✏️</div>
        <div className="floating-input-icon icon-2">📝</div>
        <div className="floating-input-icon icon-3">🔍</div>
        <div className="floating-input-icon icon-4">💭</div>
      </div>

      <div className="input-content">
        <h3 className="input-title">
          Analyze Student Statement
        </h3>
        
        <div className="input-section">
          <label htmlFor="statement" className="input-label">
            Student's Math Statement
          </label>
          <div className="textarea-container">
            <textarea
              id="statement"
              value={statement}
              onChange={(e) => setStatement(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter student's math statement or reasoning (e.g., 'I added all sides to find area', '0.355 is larger because 355 > 8')"
              className="statement-textarea"
              disabled={isLoading}
            />
            <div className="textarea-decoration"></div>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <div className="error-icon">⚠️</div>
            <div className="error-text">{error}</div>
          </div>
        )}

        <button
          onClick={handleAnalyze}
          disabled={isLoading || !statement.trim()}
          className="analyze-button"
        >
          {isLoading ? (
            <>
              <LoadingSpinner />
              <span className="button-text">Analyzing...</span>
            </>
          ) : (
            <>
              <span className="button-icon">🔍</span>
              <span className="button-text">Detect Misconceptions</span>
            </>
          )}
        </button>

        <div className="examples-section">
          <p className="examples-title">Try these examples:</p>
          <div className="examples-grid">
            {examples.map((example, index) => (
              <button
                key={index}
                onClick={() => setStatement(example)}
                disabled={isLoading}
                className="example-button"
              >
                <span className="example-text">
                  {example.slice(0, 35)}...
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentInput;