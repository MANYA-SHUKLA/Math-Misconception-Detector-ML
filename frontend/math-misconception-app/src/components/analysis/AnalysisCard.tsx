import React from 'react';
import { MisconceptionAnalysis } from '@/utils/api';
import '@/styles/components/AnalysisCard.css';

interface AnalysisCardProps {
  analysis: MisconceptionAnalysis;
}

const AnalysisCard: React.FC<AnalysisCardProps> = ({ analysis }) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'confidence-high';
    if (confidence >= 60) return 'confidence-medium';
    return 'confidence-low';
  };

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 80) return 'confidence-bg-high';
    if (confidence >= 60) return 'confidence-bg-medium';
    return 'confidence-bg-low';
  };

  return (
    <div className="analysis-card">
      {/* Floating Background Elements */}
      <div className="card-floating-elements">
        <div className="floating-icon icon-1">🧮</div>
        <div className="floating-icon icon-2">📊</div>
        <div className="floating-icon icon-3">🔍</div>
      </div>

      <div className="analysis-header">
        <div className="header-content">
          <h3 className="analysis-title">
            Misconception Analysis
          </h3>
          <p className="student-statement">"{analysis.student_statement}"</p>
        </div>
        <div className="header-decoration">
          <div className="decoration-line"></div>
        </div>
      </div>
      
      {/* Primary Misconception */}
      <div className="primary-misconception-section">
        <div className="section-header">
          <div className="section-icon">🎯</div>
          <h4 className="section-title">Primary Misconception</h4>
        </div>
        <div className="misconception-content">
          <div className="misconception-header">
            <span className="misconception-name">{analysis.primary_misconception.name}</span>
            <span className={`confidence-badge ${getConfidenceColor(analysis.primary_misconception.confidence)}`}>
              {analysis.primary_misconception.confidence}%
            </span>
          </div>
          <p className="misconception-description">{analysis.primary_misconception.description}</p>
          <div className="misconception-meta">
            <span className="meta-item">Source: {analysis.primary_misconception.source}</span>
            <span className="meta-divider">•</span>
            <span className="meta-item">Detected via: {analysis.primary_misconception.detection_source}</span>
          </div>
        </div>
      </div>

      {/* Additional Misconceptions */}
      {analysis.additional_misconceptions.length > 0 && (
        <div className="additional-misconceptions-section">
          <div className="section-header">
            <div className="section-icon">🔍</div>
            <h4 className="section-title">
              Additional Misconceptions ({analysis.additional_misconceptions.length})
            </h4>
          </div>
          <div className="additional-misconceptions-list">
            {analysis.additional_misconceptions.map((misconception, index) => (
              <div key={index} className="additional-misconception-item">
                <div className="misconception-header">
                  <span className="misconception-name">{misconception.name}</span>
                  <span className={`confidence-badge ${getConfidenceColor(misconception.confidence)}`}>
                    {misconception.confidence}%
                  </span>
                </div>
                <p className="misconception-description">{misconception.description}</p>
                <div className="misconception-meta">
                  <span className="meta-item">Detected via: {misconception.detection_source}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Learning Recommendations */}
      {analysis.learning_recommendations.length > 0 && (
        <div className="learning-recommendations-section">
          <div className="section-header">
            <div className="section-icon">💡</div>
            <h4 className="section-title">Learning Recommendations</h4>
          </div>
          <ul className="recommendations-list">
            {analysis.learning_recommendations.map((recommendation, index) => (
              <li key={index} className="recommendation-item">
                <span className="recommendation-bullet">•</span>
                <span className="recommendation-text">{recommendation}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Analysis Details */}
      <div className="analysis-details-section">
        <div className="details-grid">
          <div className="detail-item">
            <span className="detail-label">Concepts:</span>
            <span className="detail-value">{analysis.math_concepts_detected.join(', ') || 'None'}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Total Detected:</span>
            <span className="detail-value">{analysis.total_misconceptions_detected}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Quality:</span>
            <span className="detail-value quality-badge">{analysis.data_quality}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisCard;