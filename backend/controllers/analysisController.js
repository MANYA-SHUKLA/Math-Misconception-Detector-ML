const { analyzeExplanation, misconceptions, sampleStudents } = require('../utils/mockData');

const analysisController = {

  analyzeSingle: async (req, res, next) => {
    try {
      const { explanation } = req.body;

      if (!explanation || typeof explanation !== 'string') {
        return res.status(400).json({
          success: false,
          error: 'Explanation is required and must be a string'
        });
      }

   
      await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 700));

      const analysis = analyzeExplanation(explanation);

      const result = {
        student_id: Date.now(), // Generate unique ID
        question: "User Input Analysis",
        student_answer: "N/A",
        explanation: explanation,
        detected_misconception: analysis.detectedMisconception.name,
        confidence: analysis.confidence,
        description: analysis.detectedMisconception.description,
        math_concepts_detected: analysis.mathConcepts,
        student_sentiment: analysis.sentiment
      };

      res.json({
        success: true,
        data: result,
        message: 'Analysis completed successfully'
      });

    } catch (error) {
      next(error);
    }
  },

  // Analyze batch of students
  analyzeBatch: async (req, res, next) => {
    try {
      const { students } = req.body;

      if (!Array.isArray(students)) {
        return res.status(400).json({
          success: false,
          error: 'Students must be an array'
        });
      }

      const results = [];

      for (const student of students) {
        // Simulate processing each student with delay
        await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 500));

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
        message: `Batch analysis completed for ${results.length} students`
      });

    } catch (error) {
      next(error);
    }
  },

  // Get all misconceptions
  getMisconceptions: (req, res, next) => {
    try {
      res.json({
        success: true,
        data: misconceptions,
        count: misconceptions.length,
        message: 'Misconceptions retrieved successfully'
      });
    } catch (error) {
      next(error);
    }
  },

  // Get sample students
  getSampleStudents: (req, res, next) => {
    try {
      res.json({
        success: true,
        data: sampleStudents,
        count: sampleStudents.length,
        message: 'Sample students retrieved successfully'
      });
    } catch (error) {
      next(error);
    }
  },

  // Health check
  healthCheck: (req, res) => {
    res.json({
      success: true,
      message: 'Analysis controller is working',
      timestamp: new Date().toISOString(),
      version: '1.0.0'
    });
  }
};

module.exports = analysisController;