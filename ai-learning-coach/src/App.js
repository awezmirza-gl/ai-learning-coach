// App.jsx
import { useState } from "react";

const API_URL = "http://localhost:5000/api/evaluate";

// ── Inline styles as constants to keep JSX clean ──────────────────────────────
const styles = {
  root: {
    minHeight: "100vh",
    background: "#0d0f14",
    color: "#e8e6e0",
    fontFamily: "'Georgia', 'Times New Roman', serif",
    padding: "0 16px 60px",
  },
  header: {
    maxWidth: 780,
    margin: "0 auto",
    padding: "48px 0 32px",
    borderBottom: "1px solid #2a2d35",
  },
  badge: {
    display: "inline-block",
    fontSize: 11,
    letterSpacing: "0.18em",
    textTransform: "uppercase",
    color: "#7ee8a2",
    border: "1px solid #7ee8a2",
    borderRadius: 2,
    padding: "3px 10px",
    marginBottom: 16,
  },
  h1: {
    fontSize: "clamp(26px, 5vw, 42px)",
    fontWeight: 700,
    lineHeight: 1.15,
    margin: "0 0 10px",
    color: "#f5f3ee",
    letterSpacing: "-0.02em",
  },
  subtitle: {
    color: "#6b7280",
    fontSize: 15,
    margin: 0,
    lineHeight: 1.6,
  },
  main: {
    maxWidth: 780,
    margin: "0 auto",
    paddingTop: 36,
  },
  label: {
    display: "block",
    fontSize: 12,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "#9ca3af",
    marginBottom: 10,
  },
  textarea: {
    width: "100%",
    padding: "16px",
    background: "#161820",
    border: "1px solid #2a2d35",
    borderRadius: 6,
    color: "#e8e6e0",
    fontSize: 15,
    lineHeight: 1.6,
    resize: "vertical",
    boxSizing: "border-box",
    outline: "none",
    fontFamily: "inherit",
    transition: "border-color 0.2s",
  },
  btnRow: {
    display: "flex",
    alignItems: "center",
    gap: 14,
    marginTop: 14,
  },
  submitBtn: (loading) => ({
    padding: "11px 28px",
    background: loading ? "#1f3a2a" : "#7ee8a2",
    color: loading ? "#7ee8a2" : "#0d0f14",
    border: "none",
    borderRadius: 4,
    fontSize: 14,
    fontWeight: 700,
    letterSpacing: "0.05em",
    cursor: loading ? "not-allowed" : "pointer",
    transition: "all 0.2s",
    fontFamily: "inherit",
  }),
  spinner: {
    width: 18,
    height: 18,
    border: "2px solid #2a2d35",
    borderTop: "2px solid #7ee8a2",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  errorBox: {
    marginTop: 20,
    padding: "14px 18px",
    background: "#1f1215",
    border: "1px solid #7f1d1d",
    borderRadius: 6,
    color: "#fca5a5",
    fontSize: 14,
    lineHeight: 1.5,
  },
  results: {
    marginTop: 40,
    display: "flex",
    flexDirection: "column",
    gap: 24,
  },
  card: (accent) => ({
    background: "#161820",
    border: `1px solid ${accent || "#2a2d35"}`,
    borderRadius: 8,
    padding: "24px 28px",
  }),
  cardLabel: (color) => ({
    fontSize: 11,
    letterSpacing: "0.14em",
    textTransform: "uppercase",
    color: color || "#6b7280",
    marginBottom: 10,
    display: "block",
  }),
  scoreNum: (score) => ({
    fontSize: 56,
    fontWeight: 700,
    lineHeight: 1,
    color: score >= 50 ? "#7ee8a2" : "#f87171",
    letterSpacing: "-0.04em",
  }),
  scoreBar: {
    height: 6,
    background: "#2a2d35",
    borderRadius: 99,
    marginTop: 14,
    overflow: "hidden",
  },
  scoreBarFill: (score) => ({
    height: "100%",
    width: `${score}%`,
    background: score >= 50
      ? "linear-gradient(90deg, #34d399, #7ee8a2)"
      : "linear-gradient(90deg, #ef4444, #f87171)",
    borderRadius: 99,
    transition: "width 0.8s ease",
  }),
  levelPill: (level) => ({
    display: "inline-block",
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    padding: "4px 12px",
    borderRadius: 99,
    background: level === "advanced" ? "#1a2e3a" : "#2a1f0f",
    color: level === "advanced" ? "#38bdf8" : "#fb923c",
    border: `1px solid ${level === "advanced" ? "#38bdf8" : "#fb923c"}`,
    marginLeft: 10,
  }),
  tagList: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 10,
  },
  tag: (type) => ({
    fontSize: 13,
    padding: "4px 12px",
    borderRadius: 99,
    background: type === "strength" ? "#052e16" : "#2d0707",
    color: type === "strength" ? "#86efac" : "#fca5a5",
    border: `1px solid ${type === "strength" ? "#166534" : "#7f1d1d"}`,
  }),
  pre: {
    margin: 0,
    fontSize: 14,
    lineHeight: 1.75,
    color: "#d1d5db",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    fontFamily: "inherit",
  },
  divider: {
    border: "none",
    borderTop: "1px solid #2a2d35",
    margin: "16px 0",
  },
};

// ── Keyframes injected once ───────────────────────────────────────────────────
const injectKeyframes = () => {
  if (document.getElementById("__ai-coach-kf")) return;
  const style = document.createElement("style");
  style.id = "__ai-coach-kf";
  style.textContent = `
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .fade-up { animation: fadeUp 0.4s ease both; }
    textarea:focus { border-color: #7ee8a2 !important; }
  `;
  document.head.appendChild(style);
};
injectKeyframes();

// ── Helpers ───────────────────────────────────────────────────────────────────
const parseRoadmap = (raw) => {
  if (!raw) return [];
  // Try splitting on "Week N:" pattern or newlines
  const lines = raw
    .split(/\n|(?=Week \d)/i)
    .map((l) => l.trim())
    .filter(Boolean);
  return lines.length > 1 ? lines : [raw];
};

// ── Main Component ────────────────────────────────────────────────────────────
export default function App() {
  const [answers, setAnswers] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (answers.trim().length < 10) {
      setError("Please enter at least 10 characters before submitting.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers: answers.trim() }),
      });

      // Try to parse JSON even on error responses (backend sends error JSON)
      const data = await response.json().catch(() => null);

      if (!response.ok || !data) {
        let msg = `Server error (${response.status}).`;
        if (response.status === 400) {
          msg += " Your input may be too short or contain sensitive information.";
        } else if (response.status === 500) {
          msg += " The backend encountered an error. Check the logs or try again.";
        } else {
          msg += " Is the backend running on port 5000?";
        }
        if (data?.error) msg = data.error;
        setError(msg);
        return;
      }

      if (!data.success) {
        setError(data.error || "The AI backend returned an unsuccessful response.");
        return;
      }

      // Validate that we have essential fields (score can be 0, so check !== undefined)
      if (data.score === undefined || !data.level || !data.guidance || !data.roadmap) {
        setError("Backend returned incomplete response. Please try again.");
        return;
      }

      setResult(data);
    } catch (err) {
      // Network-level failure (CORS, backend not running, etc.)
      setError(
        "Cannot reach the backend at http://localhost:5000. " +
        "Make sure Flask is running (`python app.py`) and CORS is enabled."
      );
    } finally {
      setLoading(false);
    }
  };

  const roadmapSteps = result ? parseRoadmap(result.roadmap) : [];

  // Detect if guidance is a fallback (doesn't start with model error, is plain text)
  const isFallbackGuidance = result && !result.guidance.includes("[Model Error]") && result.guidance.includes("**");

  return (
    <div style={styles.root}>
      {/* ── Header ── */}
      <header style={styles.header}>
        <span style={styles.badge}>AI Learning Coach</span>
        <h1 style={styles.h1}>Your Personalized<br />Learning Architecture</h1>
        <p style={styles.subtitle}>
          Paste your quiz answers or a summary of what you know.
          The AI will score you, coach you, and build a 4-week roadmap.
        </p>
      </header>

      <main style={styles.main}>
        {/* ── Input ── */}
        <label style={styles.label} htmlFor="answers">
          Your answers / understanding
        </label>
        <textarea
          id="answers"
          rows={7}
          style={styles.textarea}
          placeholder="e.g. I understand variables and loops but I struggle with functions, recursion, and how modules work in Python..."
          value={answers}
          onChange={(e) => setAnswers(e.target.value)}
        />

        <div style={styles.btnRow}>
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={styles.submitBtn(loading)}
          >
            {loading ? "Evaluating…" : "Submit to AI Coach"}
          </button>
          {loading && <div style={styles.spinner} />}
        </div>

        {/* ── Error ── */}
        {error && (
          <div style={styles.errorBox} className="fade-up">
            ⚠ {error}
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div style={styles.results} className="fade-up">

            {/* Card 1 — Score */}
            <div style={styles.card("#2a2d35")} className="fade-up">
              <span style={styles.cardLabel("#9ca3af")}>Step 1 · Performance Score</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
                <span style={styles.scoreNum(result.score)}>{result.score}</span>
                <span style={{ color: "#6b7280", fontSize: 20 }}>/100</span>
                <span style={styles.levelPill(result.level)}>{result.level}</span>
              </div>
              <div style={styles.scoreBar}>
                <div style={styles.scoreBarFill(result.score)} />
              </div>

              {/* Strengths & Weaknesses */}
              {result.analysis?.strengths?.length > 0 && (
                <>
                  <hr style={styles.divider} />
                  <span style={styles.cardLabel("#86efac")}>Strengths</span>
                  <div style={styles.tagList}>
                    {result.analysis.strengths.map((s, i) => (
                      <span key={i} style={styles.tag("strength")}>{s}</span>
                    ))}
                  </div>
                </>
              )}
              {result.analysis?.weaknesses?.length > 0 && (
                <>
                  <hr style={styles.divider} />
                  <span style={styles.cardLabel("#fca5a5")}>Areas to Improve</span>
                  <div style={styles.tagList}>
                    {result.analysis.weaknesses.map((w, i) => (
                      <span key={i} style={styles.tag("weakness")}>{w}</span>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* Card 2 — Guidance */}
            <div
              style={styles.card(
                result.responseType === "beginner_explanation" ? "#78350f" : "#134e4a"
              )}
              className="fade-up"
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span
                  style={styles.cardLabel(
                    result.responseType === "beginner_explanation" ? "#fb923c" : "#34d399"
                  )}
                >
                  Step 2 ·{" "}
                  {result.responseType === "beginner_explanation"
                    ? "Beginner Coaching Path"
                    : "Advanced Challenge Path"}
                </span>
                {isFallbackGuidance && (
                  <span style={{ fontSize: 11, color: "#9ca3af", fontStyle: "italic" }}>
                    ℹ curated guidance
                  </span>
                )}
              </div>
              <pre style={styles.pre}>
                {result.guidance || "No guidance returned from model."}
              </pre>
            </div>

            {/* Card 3 — Roadmap */}
            <div style={styles.card("#3b1f6b")} className="fade-up">
              <span style={styles.cardLabel("#c084fc")}>Step 3 · 4-Week Learning Roadmap</span>
              <ol style={{ paddingLeft: 20, margin: 0 }}>
                {roadmapSteps.map((step, i) => (
                  <li
                    key={i}
                    style={{
                      color: "#d1d5db",
                      fontSize: 14,
                      lineHeight: 1.8,
                      marginBottom: 6,
                    }}
                  >
                    {step}
                  </li>
                ))}
              </ol>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}