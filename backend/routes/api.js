const express = require('express');
const fetch = require('node-fetch');

const router = express.Router();

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8001';

// Proxy to Python API
router.post('/analyze', async (req, res, next) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    const data = await response.json();
    res.json({
      success: true,
      data: data,
      message: 'Analysis completed successfully'
    });

  } catch (error) {
    console.error('Python API proxy error:', error);
    
    // Fallback to mock data if Python API is not available
    const { analyzeExplanation } = require('../utils/mockData');
    const analysis = analyzeExplanation(req.body.explanation);

    const result = {
      student_id: Date.now(),
      question: "User Input Analysis",
      student_answer: "N/A",
      explanation: req.body.explanation,
      detected_misconception: analysis.detectedMisconception.name,
      confidence: analysis.confidence,
      description: analysis.detectedMisconception.description,
      math_concepts_detected: analysis.mathConcepts,
      student_sentiment: analysis.sentiment
    };

    res.json({
      success: true,
      data: result,
      message: 'Analysis completed (fallback mode)'
    });
  }
});

router.post('/analyze-batch', async (req, res, next) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/analyze-batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    const data = await response.json();
    res.json({
      success: true,
      data: data.data,
      message: `Batch analysis completed for ${data.data.length} students`
    });

  } catch (error) {
    console.error('Python API proxy error:', error);
    
    // Fallback to mock data
    const { analyzeExplanation, sampleStudents } = require('../utils/mockData');
    const results = [];

    for (const student of req.body.students) {
      const analysis = analyzeExplanation(student.explanation);
      results.push({
        student_id: student.id || Date.now() + Math.random(),
        question: student.question || "Unknown Question",
        student_answer: student.answer || "N/A",
        explanation: student.explanation,
        detected_misconception: analysis.detectedMisconception.name,
        confidence: analysis.confidence,
        description: analysis.detectedMisconception.description,
        math_concepts_detected: analysis.mathConcepts,
        student_sentiment: analysis.sentiment
      });
    }

    res.json({
      success: true,
      data: results,
      message: `Batch analysis completed (fallback mode) for ${results.length} students`
    });
  }
});

// Other routes remain the same
router.get('/misconceptions', async (req, res, next) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/misconceptions`);
    
    if (response.ok) {
      const data = await response.json();
      res.json({
        success: true,
        data: data.data,
        count: data.count,
        message: 'Misconceptions retrieved successfully'
      });
    } else {
      throw new Error('Python API not available');
    }
  } catch (error) {
    // Fallback to mock data
    const { misconceptions } = require('../utils/mockData');
    res.json({
      success: true,
      data: misconceptions,
      count: misconceptions.length,
      message: 'Misconceptions retrieved (fallback mode)'
    });
  }
});

router.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'Node.js API is working',
    python_api: PYTHON_API_URL,
    timestamp: new Date().toISOString()
  });
});

module.exports = router;