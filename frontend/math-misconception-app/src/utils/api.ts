// utils/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_PYTHON_API_URL || 'http://localhost:8001';

export interface MisconceptionAnalysis {
  student_statement: string;
  primary_misconception: {
    name: string;
    confidence: number;
    description: string;
    source: string;
    match_type: string;
    detection_source: string;
    examples: string[];
  };
  additional_misconceptions: Array<{
    name: string;
    confidence: number;
    description: string;
    source: string;
    match_type: string;
    detection_source: string;
    examples: string[];
  }>;
  total_misconceptions_detected: number;
  math_concepts_detected: string[];
  nlp_patterns_detected: string[];
  learning_recommendations: string[];
  data_quality: string;
}

export const apiClient = {
  async analyzeStatement(statement: string): Promise<MisconceptionAnalysis> {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ statement }),
    });
    
    if (!response.ok) {
      throw new Error('Analysis failed');
    }
    
    const result = await response.json();
    return result;
  },

  async getMisconceptions() {
    const response = await fetch(`${API_BASE_URL}/api/misconceptions`);
    const result = await response.json();
    return result.data;
  },

  async getHealthStatus() {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return await response.json();
  }
};