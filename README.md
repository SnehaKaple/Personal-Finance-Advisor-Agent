# 💰 Personal Finance Advisor Agent

**Agentic AI Project **


---

## 📌 Problem Statement

Young working professionals in India lack access to affordable, personalised financial guidance. Generic AI chatbots hallucinate financial data — wrong tax limits, incorrect interest rates, fabricated returns — which can cause real monetary harm. This project addresses that gap with a domain-specific, self-evaluating agentic AI assistant.

---

## 🚀 What the Agent Does

- Answers questions from a **10-topic India-specific knowledge base** (budgeting, investing, mutual funds, tax saving, loans, insurance, emergency fund, FDs, goal planning, stock market)
- Routes **calculation queries** to built-in EMI, SIP, and Rule-of-72 calculators
- **Self-evaluates** every answer for confidence (0–1); auto-retries if score < 0.6
- Maintains **multi-turn conversation memory** via LangGraph MemorySaver (sliding window of 6 turns)
- Detects and persists the **user's name** for personalised replies
- Displays **route, confidence score, and source topics** per response in the UI

---

## 🗂️ Project Structure

```
personal-finance-advisor/
│
├── capstone_streamlit.py   # Streamlit UI — chat interface, sidebar, session stats
├── agent.py                # LangGraph agent — KB, 6 nodes, graph assembly
├── requirements.txt        # All Python dependencies
├── .env                    # API key config (add your GROQ_API_KEY here)
├── day13_capstone.ipynb    # Jupyter notebook — full implementation + testing
└── README.md               # This file
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Node 1: ROUTER]  ── classify: ADVICE / CALCULATE / SKIP
    │
    ├─(ADVICE / CALCULATE)──► [Node 2: RETRIEVER]   keyword scoring → top-3 KB topics
    │                                 │
    │                        [Node 3: CALCULATOR]   EMI / SIP / Rule-of-72
    │                                 │
    └─(SKIP)────────────────► [Node 4: ANSWER]      LLM response generation
                                      │
                             [Node 5: EVALUATOR]    self-score 0–1
                                      │
                         conf < 0.6 → [Node 6: RETRY] → EVALUATOR
                                      │
                                     END
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq LLaMA 3.3 70B (`llama-3.3-70b-versatile`) |
| Agent Framework | LangGraph StateGraph — 6 nodes, conditional edges, MemorySaver |
| Knowledge Base | In-memory Python dict — 10 India-specific finance topics |
| UI Framework | Streamlit — dark theme, chat interface, session stats |
| Retrieval | Keyword scoring (TF-IDF style) — no embedding model needed |
| Environment | python-dotenv — GROQ_API_KEY loaded from `.env` |
| Language | Python 3.9+ |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/personal-finance-advisor.git
cd personal-finance-advisor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Open the `.env` file and replace the placeholder with your actual key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get a **free** Groq API key at: [https://console.groq.com](https://console.groq.com)

### 4. Run the app

```bash
streamlit run capstone_streamlit.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 Features

| Feature | Description |
|---|---|
| **Agentic Routing** | LLM-free regex router classifies every query as ADVICE / CALCULATE / SKIP |
| **10-Domain KB** | India-specific content: 80C, SEBI, NSE/BSE, Post Office schemes, RBI bonds |
| **EMI Calculator** | Reducing-balance formula — principal, rate, tenure → monthly EMI + total interest |
| **SIP Calculator** | Future value formula → total invested, estimated gains, maturity value |
| **Rule of 72** | Instantly computes years to double money at a given return rate |
| **Self-Evaluation** | LLM scores own answer 0–1; auto-regenerates if confidence < 0.6 (max 2 retries) |
| **Conversation Memory** | LangGraph MemorySaver — coherent multi-turn conversations |
| **Transparent UI** | Route badge + colour-coded confidence + source topics per response |
| **Session Stats** | Live total queries + average confidence in sidebar |
| **Error Banner** | Clear setup guidance if GROQ_API_KEY is missing |

---

## 💬 Sample Questions

| Category | Example Query |
|---|---|
| Budgeting | "What is the 50-30-20 budgeting rule?" |
| Investing | "How should I start investing with Rs 5000/month?" |
| Mutual Funds | "How does SIP and rupee cost averaging work?" |
| Tax Saving | "What are the best 80C tax saving options?" |
| Calculator | "Calculate EMI for Rs 10 lakh at 8% for 5 years" |
| Calculator | "SIP of Rs 3000 at 12% for 15 years — what will I get?" |
| Insurance | "Difference between term and endowment insurance?" |
| Emergency Fund | "How much emergency fund should I maintain?" |

---

## 🔍 Agent Routing Logic

| Route | Trigger | Flow |
|---|---|---|
| `ADVICE` | Finance questions without numbers | Router → Retriever → Calculator → Answer → Evaluator |
| `CALCULATE` | Queries with numbers + calc keywords | Router → Retriever → Calculator (computes) → Answer → Evaluator |
| `SKIP` | Greetings, chit-chat, name intro | Router → Answer (canned response, no LLM call) |

---

## 📈 Confidence Scoring

| Score | Colour | Meaning |
|---|---|---|
| ≥ 0.75 | 🟢 Green | High confidence — well-grounded answer |
| 0.50 – 0.74 | 🟡 Yellow | Moderate confidence |
| < 0.50 | 🔴 Red | Low confidence — may have been retried |

Answers below **0.60** are automatically regenerated (up to **2 retries**).

---

## 🔮 Future Improvements

- **Vector Store** (ChromaDB / pgvector) for semantic retrieval
- **Live Market Data** via NSE/BSE or Yahoo Finance APIs
- **Portfolio Tracker** with rebalancing suggestions
- **Multi-language Support** (Hindi, Tamil, Bengali)
- **Persistent User Profiles** across sessions
- **Goal Progress Dashboard** with projected timelines
- **SEBI Advisor Referral** for out-of-scope queries

---
