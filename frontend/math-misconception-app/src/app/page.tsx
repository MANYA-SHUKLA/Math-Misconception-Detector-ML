'use client';

import React from 'react';
import Link from 'next/link';
import './HomePage.css';

export default function HomePage() {
  return (
    <div className="homepage-container">
      <div className="main-content">
        {/* Header */}
        <div className="header-section">
          <div className="logo-header">
            <MathLogo className="main-logo" />
            <h1 className="main-title">
              Math Misconception Detector
            </h1>
          </div>
          <p className="subtitle">
            AI-powered system to identify and analyze common mathematical misconceptions 
            in student explanations using advanced Natural Language Processing.
          </p>
        </div>

        {/* Features Grid */}
        <div className="features-grid">
          <div className="feature-card">
            <AIIcon className="feature-icon" />
            <h3 className="feature-title">AI-Powered Analysis</h3>
            <p className="feature-description">
              Uses advanced Sentence Transformers and NLP to understand student reasoning patterns
            </p>
          </div>

          <div className="feature-card">
            <RealTimeIcon className="feature-icon" />
            <h3 className="feature-title">Real-time Detection</h3>
            <p className="feature-description">
              Instantly identifies common math misconceptions with confidence scoring
            </p>
          </div>

          <div className="feature-card">
            <InsightIcon className="feature-icon" />
            <h3 className="feature-title">Educational Insights</h3>
            <p className="feature-description">
              Provides detailed feedback and suggestions for targeted learning interventions
            </p>
          </div>
        </div>

        {/* CTA Section - DO NOT REMOVE */}
        <div className="cta-section">
          <div className="cta-card">
            <h2 className="cta-title">
              Ready to Analyze Student Thinking?
            </h2>
            <p className="cta-description">
              Start detecting mathematical misconceptions and gain insights into student learning patterns.
            </p>
            <div className="cta-buttons">
              <Link
                href="/analyze"
                className="cta-button primary"
              >
                Start Analyzing
              </Link>
             
            </div>
          </div>
        </div>

        {/* Tech Stack */}
        <div className="tech-stack-section">
          <h3 className="tech-stack-title">Powered By</h3>
          <div className="tech-stack-items">
            <div className="tech-item">
              <NextJSIcon className="tech-icon" />
              <span>Next.js 16+</span>
            </div>
            <div className="tech-item">
              <NLPIcon className="tech-icon" />
              <span>AI & NLP</span>
            </div>
            <div className="tech-item">
              <TailwindIcon className="tech-icon" />
              <span>Tailwind CSS</span>
            </div>
            <div className="tech-item">
              <PythonIcon className="tech-icon" />
              <span>Python Backend</span>
            </div>
            <div className="tech-item">
              <NodeJSIcon className="tech-icon" />
              <span>Node.js Express</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
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

// SVG Components
const MathLogo = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 100 100" fill="none">
    <circle cx="50" cy="50" r="45" fill="#4F46E5" opacity="0.1"/>
    <path d="M30 40L50 60L70 40" stroke="#4F46E5" strokeWidth="4" strokeLinecap="round"/>
    <path d="M40 30L40 70" stroke="#4F46E5" strokeWidth="4" strokeLinecap="round"/>
    <path d="M60 30L60 70" stroke="#4F46E5" strokeWidth="4" strokeLinecap="round"/>
    <circle cx="50" cy="50" r="40" stroke="#4F46E5" strokeWidth="2" fill="none"/>
  </svg>
);

const AIIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <path d="M12 2L14 7H18L15 10L17 15L12 12L7 15L9 10L6 7H10L12 2Z" fill="#4F46E5"/>
    <path d="M4 20L6 16H10L8 20H4Z" fill="#4F46E5" opacity="0.7"/>
    <path d="M20 20H16L14 16H18L20 20Z" fill="#4F46E5" opacity="0.7"/>
  </svg>
);

const RealTimeIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="#4F46E5" strokeWidth="2"/>
    <path d="M12 6V12L16 14" stroke="#4F46E5" strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 22C14 22 16 21 18 19" stroke="#4F46E5" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const InsightIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#4F46E5" strokeWidth="2"/>
    <path d="M12 16L12 8" stroke="#4F46E5" strokeWidth="2" strokeLinecap="round"/>
    <path d="M8 12L16 12" stroke="#4F46E5" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const NextJSIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
    <path d="M12 16L16 12L12 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M8 12H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const NLPIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <path d="M8 12H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 8V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M3 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M19 12H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 3V5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 19V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const TailwindIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <path d="M12 6C9.333 6 7.333 7.333 6 9C6.667 6.667 8.333 5.333 10.667 5C12.667 4.667 14.333 5.333 15.333 6.667C16.333 8 17 9.667 18 11C20.667 14.333 22.667 15 24 15C21.333 15 19.333 13.667 18 12C18.667 14.333 16.667 15.667 14.333 16C12.333 16.333 10.667 15.667 9.667 14.333C8.667 13 8 11.333 7 10C4.333 6.667 2.333 6 1 6H12Z" fill="currentColor"/>
  </svg>
);

const PythonIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <path d="M12 2C8.5 2 7 3.5 7 6V7H12V8H5C2.5 8 2 9.5 2 12C2 14.5 3.5 16 6 16H8V13C8 10.5 10.5 8 13 8H16C18.5 8 19 6.5 19 4C19 1.5 17.5 2 15 2H12Z" fill="currentColor"/>
    <path d="M12 22C15.5 22 17 20.5 17 18V17H12V16H19C21.5 16 22 14.5 22 12C22 9.5 20.5 8 18 8H16V11C16 13.5 13.5 16 11 16H8C5.5 16 5 17.5 5 20C5 22.5 6.5 22 9 22H12Z" fill="currentColor"/>
  </svg>
);

const NodeJSIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
    <path d="M12 8L16 12L12 16L8 12L12 8Z" stroke="currentColor" strokeWidth="2"/>
    <circle cx="12" cy="12" r="2" fill="currentColor"/>
  </svg>
);