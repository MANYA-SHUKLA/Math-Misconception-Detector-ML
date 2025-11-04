'use client';

import React, { useState } from 'react';
import StudentInput from '@/components/students/StudentInput';
import AnalysisCard from '@/components/analysis/AnalysisCard';
import { MisconceptionAnalysis } from '@/utils/api';
import '@/styles/pages/AnalyzePage.css';

export default function AnalyzePage() {
  const [analysisResults, setAnalysisResults] = useState<MisconceptionAnalysis[]>([]);

  const handleAnalysisComplete = (result: MisconceptionAnalysis) => {
    setAnalysisResults(prev => [result, ...prev]);
  };

  const clearResults = () => {
    setAnalysisResults([]);
  };

  return (
    <div className="analyze-page-container">
      {/* Floating Background Elements */}
      <div className="floating-elements">
        <div className="floating-math-icon icon-1">∫</div>
        <div className="floating-math-icon icon-2">π</div>
        <div className="floating-math-icon icon-3">∞</div>
        <div className="floating-math-icon icon-4">Δ</div>
        <div className="floating-math-icon icon-5">∑</div>
        <div className="floating-math-icon icon-6">θ</div>
      </div>

      <div className="analyze-content">
        {/* Header */}
        <div className="analyze-header">
          <h1 className="analyze-title">
            Analyze Student Explanations
          </h1>
          <p className="analyze-subtitle">
            Enter student math explanations to detect common misconceptions using AI
          </p>
        </div>

        <div className="analyze-grid">
          {/* Input Section */}
          <div className="input-section">
            <StudentInput 
              onAnalysisComplete={handleAnalysisComplete}
            />
            
            {/* Results Summary */}
            {analysisResults.length > 0 && (
              <div className="summary-card">
                <h3 className="summary-title">
                  Analysis Summary
                </h3>
                <div className="summary-stats">
                  <div className="stat-item">
                    <span className="stat-label">Total Analyses:</span>
                    <span className="stat-value">{analysisResults.length}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Avg Confidence:</span>
                    <span className="stat-value confidence">
                      {Math.round(
                        analysisResults.reduce((acc, curr) => acc + (curr.primary_misconception?.confidence ?? 0), 0) /
                        analysisResults.length
                      )}%
                    </span>
                  </div>
                </div>
                <button
                  onClick={clearResults}
                  className="clear-button secondary"
                >
                  Clear All Results
                </button>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="results-section">
            {analysisResults.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <h3 className="empty-title">
                  No Analyses Yet
                </h3>
                <p className="empty-description">
                  Enter a student explanation on the left to start detecting math misconceptions.
                </p>
              </div>
            ) : (
              <div className="results-container">
                <div className="results-header">
                  <h2 className="results-title">
                    Analysis Results ({analysisResults.length})
                  </h2>
                  <button
                    onClick={clearResults}
                    className="clear-button text-only"
                  >
                    Clear All
                  </button>
                </div>
                
                <div className="results-list">
                  {analysisResults.map((result, index) => (
                    <AnalysisCard key={index} analysis={result} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      <footer className="footer">
        <div className="footer-content">
          <p className="footer-text">
            Submitted to: ML Prerna Ma'am | 
            Tech Stack: NLP, Machine Learning, Next.js Frontend, Node.js Express Backend | 
            Made by Manya Shukla 2025
          </p>
        </div>
      </footer>
    </div>
  );
}