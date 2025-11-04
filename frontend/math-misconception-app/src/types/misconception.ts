// types/misconception.ts
export interface AnalysisResult {
  student_id: number;
  question: string;
  student_answer: string;
  explanation: string;
  detected_misconception: string;
  confidence: number;
  description: string;
  source: string;
  math_concepts_detected: string[];
  nlp_patterns_detected: string[];
  data_quality: string;
  all_matches: Array<{
    misconception: string;
    confidence: number;
    source: string;
  }>;
  // Optional field from your AnalysisCard
  student_sentiment?: string;
}

export interface Misconception {
  code: string;
  name: string;
  description: string;
  examples: string[];
  subject: string;
  source: string;
}