
# Vigilis – Tiered Log Classification

Vigilis turns the chaos of system logs into structured, actionable intelligence. Instead of relying on brittle keyword searches or expensive commercial tools, it uses a smart tiered approach—combining specific rules, semantic understanding, and generative AI—to classify logs with both speed and precision.

## Why Vigilis?

Modern applications generate gigabytes of logs. Parsing them manually is impossible, and traditional regex rules break easily. Vigilis solves this by layering three technologies:

1.  **Regex (The Speed Layer):** Instantly catches known, repetitive patterns.
2.  **BERT (The Semantic Layer):** Understands the *meaning* of logs, not just keywords, catching variations that regex misses.
3.  **LLM (The Intelligence Layer):** Leverages a Large Language Model (Google Gemini via LangChain) to reason through ambiguous, rare, or complex log messages just like a human engineer would.

## Features

- **Hybrid Intelligence**: Balances cost and accuracy by using the cheapest effective model first (Regex → BERT → LLM).
- **Modern UI**: A sleek, dark-mode web interface for uploading bulk CSVs and visualizing the results instantly.
- **API-First Design**: Integrate directly into your specialized monitoring pipelines with simple REST endpoints.
- **Observable**: Built-in Prometheus metrics to track classification distribution, latency, and error rates.

## Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend** | Python 3.12, FastAPI | The core orchestration engine. |
| **Logic** | Pandas, NumPy | Efficient data handling and transformation. |
| **AI/ML** | SentenceTransformers (BERT), Scikit-learn | Semantic embedding and classification. |
| **GenAI** | Google Gemini (Gemini 3 Pro Preview) | High-speed inference for complex reasoning. |
| **Frontend** | Jinja2, Custom CSS | A responsive, no-framework web UI. |
| **Ops** | UV, Prometheus | Modern dependency management and monitoring. |

## How It Works

Imagine a log message entering Vigilis: _"Database connection refused at 10.0.0.5"_

1.  **Check 1 (Regex)**: Does this match a strict rule like `^Database connection.*`? If yes, label it **Database Error**. Efficiency: 100%.
2.  **Check 2 (BERT)**: If regex fails, we convert the text to numbers (embeddings) and ask our trained model. It recognizes the semantic similarity to other connection errors. Label: **Database Error**.
3.  **Check 3 (LLM)**: If the message is weird, like _"The oracle refuses to speak,"_ the LLM analyzes the context and determines it's likely a **Service Outage**.

## Directory Structure

```text
Vigilis/
├── app/                  # The web application
│   ├── templates/        # HTML interfaces
│   └── static/           # CSS and assets
├── processors/           # The brains of the operation
│   ├── regex_processing.py
│   ├── bert_processing.py
│   └── llm_processing.py
├── models/               # Serialized BERT models
├── notebooks/            # Research and training grounds
├── tests/                # Unit tests ensuring reliability
└── utils/                # shared helpers
```

## Getting Started

### 1. Setup

Clone the repo and install dependencies using `uv` (the ultra-fast Python package installer):

```bash
git clone https://github.com/your-username/Vigilis.git
cd Vigilis
uv sync
```

### 2. Configure

Vigilis needs a brain for the hard stuff. Get a free API key from [Google AI Studio](https://aistudio.google.com/) and set it up:

Create a `.env` file:
```env
GOOGLE_API_KEY="your_api_key_here"
```

### 3. Run

Launch the web interface:

```bash
uv run uvicorn app.app:app --reload
```

Then open **http://localhost:8000** to see Vigilis in action.

## Testing

We believe in reliable code. Run the full suite to verify everything is green:

```bash
uv run -m unittest discover tests
```
