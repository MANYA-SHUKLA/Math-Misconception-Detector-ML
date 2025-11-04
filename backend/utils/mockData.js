const misconceptions = [
  {
    code: "DECIMAL_ERROR",
    name: "Decimal Comparison Error",
    description: "Student compares decimals like whole numbers",
    examples: ["Thinks 0.355 > 0.8 because 355 > 8"],
    subject: "number_sense"
  },
  {
    code: "AREA_PERIMETER",
    name: "Area vs Perimeter Confusion",
    description: "Student confuses area and perimeter formulas",
    examples: ["Adds all sides to find area instead of multiplying"],
    subject: "geometry"
  },
  {
    code: "ORDER_OPS",
    name: "Order of Operations Error",
    description: "Student doesn't follow PEMDAS rules",
    examples: ["Calculates 3+4×2 as 14 instead of 11"],
    subject: "arithmetic"
  },
  {
    code: "SLOPE_INTERCEPT",
    name: "Slope-Intercept Confusion",
    description: "Student thinks y-intercept is the slope",
    examples: ["In y=2x+5, thinks slope is 5 instead of 2"],
    subject: "algebra"
  },
  {
    code: "FRACTION_COMPARISON",
    name: "Fraction Comparison Error",
    description: "Student incorrectly compares fractions",
    examples: ["Thinks 1/4 > 1/2 because 4 > 2"],
    subject: "fractions"
  },
  {
    code: "PERCENTAGE_ERROR",
    name: "Percentage Conversion Error",
    description: "Student incorrectly converts percentages",
    examples: ["Thinks 0.5 = 5% instead of 50%"],
    subject: "number_sense"
  }
];

const sampleStudents = [
  {
    id: 1,
    question: "Which is larger: 0.355 or 0.8?",
    answer: "0.355",
    explanation: "355 is greater than 8, so 0.355 is larger",
    correct_answer: "0.8",
    subject: "number_sense"
  },
  {
    id: 2,
    question: "Find area of rectangle: length=5cm, width=3cm",
    answer: "16cm",
    explanation: "I added all sides: 5+5+3+3=16",
    correct_answer: "15cm²",
    subject: "geometry"
  },
  {
    id: 3,
    question: "Calculate: 3 + 4 × 2",
    answer: "14",
    explanation: "First I added 3+4=7, then multiplied by 2",
    correct_answer: "11",
    subject: "arithmetic"
  },
  {
    id: 4,
    question: "What is the slope of y = 2x + 5?",
    answer: "5",
    explanation: "The number without x is the slope",
    correct_answer: "2",
    subject: "algebra"
  }
];

// Mock AI analysis function
function analyzeExplanation(explanation) {
  const explanationLower = explanation.toLowerCase();
  
  // Simple pattern matching for demonstration
  let detectedMisconception = null;
  let confidence = 0;
  let mathConcepts = [];
  let sentiment = "neutral";

  // Pattern matching logic
  if (explanationLower.includes('add') && explanationLower.includes('side') && 
      (explanationLower.includes('area') || explanationLower.includes('find'))) {
    detectedMisconception = misconceptions.find(m => m.code === "AREA_PERIMETER");
    confidence = 85 + Math.random() * 10;
    mathConcepts = ['area_perimeter', 'measurement'];
  } 
  else if ((explanationLower.includes('number without x') || explanationLower.includes('constant')) && 
           explanationLower.includes('slope')) {
    detectedMisconception = misconceptions.find(m => m.code === "SLOPE_INTERCEPT");
    confidence = 90 + Math.random() * 5;
    mathConcepts = ['slope_intercept', 'linear_equations'];
  }
  else if (explanationLower.includes('add') && explanationLower.includes('multipl') && 
           explanationLower.includes('first')) {
    detectedMisconception = misconceptions.find(m => m.code === "ORDER_OPS");
    confidence = 80 + Math.random() * 15;
    mathConcepts = ['order_operations', 'arithmetic'];
  }
  else if (explanationLower.match(/\d+\.\d+/) && 
          (explanationLower.includes('bigger') || explanationLower.includes('larger') || explanationLower.includes('greater'))) {
    detectedMisconception = misconceptions.find(m => m.code === "DECIMAL_ERROR");
    confidence = 75 + Math.random() * 20;
    mathConcepts = ['decimal_operation', 'number_comparison'];
  }
  else if (explanationLower.includes('fraction') || explanationLower.includes('1/')) {
    detectedMisconception = misconceptions.find(m => m.code === "FRACTION_COMPARISON");
    confidence = 70 + Math.random() * 25;
    mathConcepts = ['fractions', 'number_sense'];
  }
  else if (explanationLower.includes('%') || explanationLower.includes('percent')) {
    detectedMisconception = misconceptions.find(m => m.code === "PERCENTAGE_ERROR");
    confidence = 65 + Math.random() * 30;
    mathConcepts = ['percentage', 'conversion'];
  }
  else {
    // Fallback: find the best match based on keyword similarity
    const keywordMatches = misconceptions.map(misconception => {
      const keywordCount = misconception.examples[0].toLowerCase()
        .split(' ')
        .filter(word => explanationLower.includes(word))
        .length;
      return { misconception, score: keywordCount };
    }).filter(match => match.score > 0);

    if (keywordMatches.length > 0) {
      const bestMatch = keywordMatches.reduce((best, current) => 
        current.score > best.score ? current : best
      );
      detectedMisconception = bestMatch.misconception;
      confidence = 50 + (bestMatch.score * 10);
      mathConcepts = ['general_math'];
    }
  }

  // Default if no match found
  if (!detectedMisconception) {
    detectedMisconception = misconceptions[0];
    confidence = 40 + Math.random() * 20;
    mathConcepts = ['basic_math'];
  }

  // Sentiment analysis (simple)
  if (explanationLower.includes('sure') || explanationLower.includes('think') || explanationLower.includes('maybe')) {
    sentiment = "unsure";
  } else if (explanationLower.includes('know') || explanationLower.includes('certain') || explanationLower.includes('definitely')) {
    sentiment = "confident";
  }

  return {
    detectedMisconception,
    confidence: Math.min(95, Math.max(40, confidence)), // Clamp between 40-95
    mathConcepts,
    sentiment
  };
}

module.exports = {
  misconceptions,
  sampleStudents,
  analyzeExplanation
};