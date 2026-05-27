# AI Learning Coach

## Project Overview

The AI Learning Coach is a full-stack web application designed to provide a personalized learning experience for Python programming. It uses a series of AI-powered calls to evaluate a user's knowledge, provide tailored feedback, and generate a custom learning plan. The project consists of a React frontend that communicates with a Flask backend, which in turn leverages HuggingFace models for its AI capabilities.

## Features

-   **AI-Powered Evaluation**: Analyzes a user's summary of their Python knowledge to gauge their proficiency.
-   **Dynamic Scoring and Leveling**: Scores the user on a scale of 0-100 and classifies them as "beginner" or "advanced."
-   **Personalized Guidance**:
    -   **Beginners** receive clear, simple explanations of fundamental concepts.
    -   **Advanced** users are given a set of coding challenges covering data structures, REST APIs, and system design.
-   **Custom Learning Roadmap**: Generates a structured 4-week learning roadmap based on the user's score and level.
-   **Adaptive Quiz**: Offers a quiz with questions that adapt to the user's assessed skill level (beginner or advanced).
-   **Robust Safety Measures**: Implements rate limiting, PII (Personally Identifiable Information) detection, input sanitization, and a unique LLM-based "judge" to ensure AI-generated content is safe and appropriate for users.
-   **Modern UI**: A sleek, responsive, dark-mode interface built with React provides a polished user experience.

## Setup Instructions

To run this project locally, you will need to set up both the backend and frontend services.

### Backend (Flask)

1.  **Navigate to the root directory:**
    ```bash
    cd /path/to/ai-learning-coach
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    The project uses Flask, OpenAI's client library, and a few other packages. Install them using pip:
    ```bash
    pip install flask flask_cors flask_limiter openai python-dotenv
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your HuggingFace API token. You can get a token from the [HuggingFace website](https://huggingface.co/settings/tokens).
    ```
    HF_TOKEN="your_huggingface_token_here"
    ```

5.  **Run the Flask server:**
    ```bash
    python app.py
    ```
    The backend will be running at `http://localhost:5000`.

### Frontend (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd /path/to/ai-learning-coach/ai-learning-coach
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

3.  **Start the React development server:**
    ```bash
    npm start
    ```
    The frontend will open automatically in your browser at `http://localhost:3000`.

## Architecture & Workflow

The application follows a standard client-server architecture.

### Frontend (React)

-   The UI is built as a single-page application using React, located in the `ai-learning-coach/` sub-directory.
-   The main component, `App.js`, manages the user input, application state, and results display.
-   When a user submits their text, the frontend makes a POST request to the `/api/evaluate` endpoint on the Flask backend.
-   Based on the AI's evaluation, it may also fetch an adaptive quiz from the `/api/quiz` endpoint.
-   Results, including the score, analysis, guidance, and roadmap, are dynamically rendered for the user.

### Backend (Flask)

-   The backend is a Python Flask server defined in `app.py`.
-   It exposes two primary endpoints: `/api/evaluate` for processing user input and `/api/quiz` for serving quiz questions.
-   The core of the backend is a three-call LLM (Large Language Model) pipeline that uses the HuggingFace Inference Router via its OpenAI-compatible API.

#### The 3-Call LLM Pipeline:

1.  **Call 1: Analysis (Mistral)**
    -   The user's input is sent to a Mistral model to be analyzed.
    -   The AI returns a structured JSON object containing a `score` (0-100), an assessed `level` ("beginner" or "advanced"), and lists of `strengths` and `weaknesses`.

2.  **Call 2: Guidance (Mistral)**
    -   Based on the `level` from the first call, a second call is made to generate personalized guidance.
    -   If the user is a beginner, it provides easy-to-understand explanations.
    -   If the user is advanced, it generates a set of coding challenges.

3.  **Call 3: Roadmap (Mistral)**
    -   Using the `score` and `level`, a third call to the Mistral model generates a personalized 4-week learning plan, with weekly goals and tasks.

## AI Capabilities Used

-   **HuggingFace Inference Router**: The application uses HuggingFace models through their high-performance, OpenAI-compatible API, allowing for easy integration.
-   **Mistral Models**: The project primarily leverages `mistralai/Mistral-7B-Instruct-v0.2` for all its core AI tasks due to its strong instruction-following capabilities.
-   **Multi-call LLM Chaining**: Instead of relying on a single, complex prompt, the backend chains multiple, focused LLM calls. The output of one call informs the input of the next, leading to a more accurate and context-aware final result.
-   **LLM as a Judge**: A novel safety feature where one LLM is used to evaluate the generated content from another. Before any AI-generated text is sent to the user, it is first sent to a "judge" model that checks it for safety, appropriateness, and harmful content. This significantly reduces the risk of exposing users to unsafe or low-quality output.

## Challenges Faced

-   **Ensuring Reliable AI Output**: LLMs can sometimes produce unpredictable or improperly formatted responses. This was addressed by implementing strict system prompts that command the model to return `ONLY` valid JSON, along with fallback mechanisms and robust parsing functions (`safe_extract_analysis`) on the backend.
-   **Content Safety and Moderation**: To prevent harmful or inappropriate content from reaching the user, an "LLM as a Judge" pattern was implemented. This adds a layer of AI-powered moderation that automatically flags and replaces unsafe content.
-   **Preventing Prompt Injection and Misuse**: Security was a key consideration. The system includes a `sanitize` function to strip out characters commonly used in prompt injection attacks and a `MAX_ANSWER_LENGTH` to prevent abuse.
-   **Protecting User Privacy**: An explicit check for Personally Identifiable Information (`has_personal_info`) was created to detect and block submissions containing sensitive data like emails, phone numbers, or financial details.

## Future Improvements

-   **Persistent User Data**: Integrate a database (e.g., SQLite or PostgreSQL) to store user history, track progress over time, and save learning roadmaps.
-   **User Authentication**: Implement user accounts to provide a more personalized and continuous learning journey.
-   **Expanded Content**: Add more subjects beyond Python and increase the number and variety of quiz questions.
-   **Advanced State Management**: As the frontend grows, incorporate a dedicated state management library like Redux or Zustand.
-   **Configuration Management**: Move model names and other settings from hardcoded constants to a more flexible configuration file or environment variables.
