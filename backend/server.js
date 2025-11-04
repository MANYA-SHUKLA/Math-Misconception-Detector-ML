const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
require('dotenv').config();

const app = express();

app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

const PYTHON_API = 'http://localhost:8001'; // Updated to match frontend .env

// Health check
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    message: 'Math Misconception Detector API is running',
    timestamp: new Date().toISOString()
  });
});

// Proxy to Python API
app.post('/api/analyze', async (req, res) => {
  try {
    const response = await fetch(`${PYTHON_API}/api/analyze`, {
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
    res.status(500).json({
      success: false,
      error: 'Analysis service unavailable'
    });
  }
});

// Proxy misconceptions
app.get('/api/misconceptions', async (req, res) => {
  try {
    const response = await fetch(`${PYTHON_API}/api/misconceptions`);
    
    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    const data = await response.json();
    res.json({
      success: true,
      data: data.data,
      count: data.count
    });

  } catch (error) {
    console.error('Python API proxy error:', error);
    res.status(500).json({
      success: false,
      error: 'Misconceptions service unavailable'
    });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Node.js server running on port ${PORT}`);
  console.log(`🔗 Proxying to Python API: ${PYTHON_API}`);
});
