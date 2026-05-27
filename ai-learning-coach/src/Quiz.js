// ai-learning-coach/src/Quiz.js
import { useState } from "react";

const styles = {
  card: {
    background: "#161820",
    border: `1px solid #3b1f6b`,
    borderRadius: 8,
    padding: "24px 28px",
    marginTop: "20px"
  },
  cardLabel: {
    fontSize: 11,
    letterSpacing: "0.14em",
    textTransform: "uppercase",
    color: "#c084fc",
    marginBottom: 10,
    display: "block",
  },
  question: {
    fontSize: "18px",
    fontWeight: "bold",
    marginBottom: "20px"
  },
  options: {
    display: "flex",
    flexDirection: "column",
    gap: "10px"
  },
  option: {
    padding: "10px",
    border: "1px solid #2a2d35",
    borderRadius: "4px",
    cursor: "pointer"
  },
  selectedOption: {
    padding: "10px",
    border: "1px solid #7ee8a2",
    borderRadius: "4px",
    cursor: "pointer",
    background: "#1f3a2a"
  },
  navigation: {
    display: "flex",
    justifyContent: "space-between",
    marginTop: "20px"
  },
  button: {
    padding: "11px 28px",
    background: "#7ee8a2",
    color: "#0d0f14",
    border: "none",
    borderRadius: 4,
    fontSize: 14,
    fontWeight: 700,
    letterSpacing: "0.05em",
    cursor: "pointer",
    fontFamily: "inherit",
  },
  disabledButton: {
    padding: "11px 28px",
    background: "#1f3a2a",
    color: "#7ee8a2",
    border: "none",
    borderRadius: 4,
    fontSize: 14,
    fontWeight: 700,
    letterSpacing: "0.05em",
    cursor: "not-allowed",
    fontFamily: "inherit",
  },
  score: {
    fontSize: "24px",
    fontWeight: "bold",
    marginTop: "20px"
  }
};

export default function Quiz({ questions }) {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [showScore, setShowScore] = useState(false);

  const handleOptionSelect = (option) => {
    setUserAnswers({
      ...userAnswers,
      [currentQuestionIndex]: option
    });
  };

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmit = () => {
    setShowScore(true);
  };

  const calculateScore = () => {
    let score = 0;
    questions.forEach((question, index) => {
      if (userAnswers[index] === question.correct_answer) {
        score++;
      }
    });
    return score;
  };

  if (showScore) {
    return (
      <div style={styles.card}>
        <span style={styles.cardLabel}>Quiz Results</span>
        <div style={styles.score}>
          Your score: {calculateScore()} / {questions.length}
        </div>
      </div>
    );
  }

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <div style={styles.card}>
        <span style={styles.cardLabel}>Adaptive Quiz</span>
        <div style={styles.question}>{currentQuestion.question}</div>
        <div style={styles.options}>
            {currentQuestion.options.map((option, index) => (
            <div
                key={index}
                style={userAnswers[currentQuestionIndex] === option ? styles.selectedOption : styles.option}
                onClick={() => handleOptionSelect(option)}
            >
                {option}
            </div>
            ))}
        </div>
        <div style={styles.navigation}>
            <button 
                onClick={handlePrevious} 
                disabled={currentQuestionIndex === 0}
                style={currentQuestionIndex === 0 ? styles.disabledButton : styles.button}
            >
                Previous
            </button>
            {currentQuestionIndex < questions.length - 1 ? (
                <button 
                    onClick={handleNext} 
                    style={styles.button}
                >
                    Next
                </button>
            ) : (
                <button 
                    onClick={handleSubmit} 
                    style={styles.button}
                >
                    Submit
                </button>
            )}
        </div>
    </div>
  );
}
