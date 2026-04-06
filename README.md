# 🎓 Multi-Agent AI Academic Assistant

A production-ready educational AI system where three specialised agents collaborate in real time to provide personalised tutoring, learning analytics, and student support — powered by **CrewAI**, **LangChain**, **GPT-4**, and **ChromaDB**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Personalised Tutoring** | GPT-4 searches your uploaded course PDFs via RAG and adapts explanations to each student's performance level |
| **Learning Analytics** | Tracks scores, completion rates, trends, strengths & weaknesses from persistent JSON profiles |
| **Admin Support** | Instant answers on admissions, fees, schedules, and academic policies |
| **Progress Reports** | AI-generated 4-week study plans with SMART goals and motivational insights |
| **PDF RAG** | Upload lecture notes/textbooks; ChromaDB indexes them for semantic search |
| **Real-time Agent Log** | Expandable panel shows every agent thought, action, and observation |
| **Demo Mode** | Full dashboard preview without an API key |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                 │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │ Tutoring │   │Admin Support│   │ Progress Report  │  │
│  └────┬─────┘   └──────┬──────┘   └────────┬─────────┘  │
└───────┼────────────────┼───────────────────┼────────────┘
        │                │                   │
┌───────▼────────────────▼───────────────────▼────────────┐
│               AcademicAssistantCrew (crew.py)            │
│                   CrewAI · Sequential                     │
│  ┌──────────────┐ ┌─────────────────┐ ┌───────────────┐  │
│  │ Tutor Agent  │ │Analytics Agent  │ │ Support Agent │  │
│  │   GPT-4 0.7  │ │   GPT-4  0.3   │ │ GPT-3.5  0.5  │  │
│  └──────┬───────┘ └────────┬────────┘ └──────┬────────┘  │
└─────────┼──────────────────┼─────────────────┼───────────┘
          │                  │                  │
┌─────────▼──────────────────▼─────────────────▼───────────┐
│                         Tools Layer                        │
│  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │ DocumentSearch   │  │   PerformanceAnalysis        │   │
│  │ (ChromaDB + RAG) │  │   ProgressTracker            │   │
│  │ PolicySearch     │  │   (JSON student profiles)    │   │
│  └──────────────────┘  └──────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Agent Collaboration Flow

**Tutoring Request:**
```
User Question
    │
    ▼
Analytics Agent ──── analyses student profile ──→ learning level context
    │
    ▼
Tutor Agent ──── searches PDFs ──── crafts explanation ──→ Response + 3 Practice Qs
```

**Progress Report:**
```
Generate Report
    │
    ▼
Analytics Agent ──── deep performance analysis ──→ metrics + trend report
    │
    ▼
Tutor Agent ──── builds 4-week personalised learning plan ──→ Full Report
```

---

## 🚀 Quick Start

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd multi-agent-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env            # or edit .env directly
```

Open `.env` and replace the placeholder:

```env
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-4
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |
| `OPENAI_MODEL` | Optional | Model override (default: `gpt-4`) |

---

## 📂 Project Structure

```
multi-agent-ai-assistant/
├── app.py                      # Streamlit web application
├── crew.py                     # AcademicAssistantCrew orchestration
├── requirements.txt
├── .env                        # API keys (not committed)
│
├── agents/
│   ├── __init__.py
│   ├── tutor_agent.py          # GPT-4 teaching agent
│   ├── analytics_agent.py      # GPT-4 data analytics agent
│   └── support_agent.py        # GPT-3.5-turbo admin agent
│
├── tools/
│   ├── __init__.py
│   ├── document_tools.py       # DocumentSearchTool + PolicySearchTool + indexer
│   └── analytics_tools.py      # PerformanceAnalysisTool + ProgressTrackerTool
│
└── data/
    ├── documents/              # Uploaded PDF course materials
    ├── student_data/           # Student JSON profiles
    │   └── student_001.json    # Demo student (Alex Johnson)
    └── chroma_db/              # ChromaDB vector store (auto-created)
```

---

## 💡 Usage Examples

### Example 1 — Tutoring

1. Select **"📚 Tutoring"** mode.
2. Enter Student ID `student_001`.
3. Upload a course PDF (optional).
4. Ask: *"Explain photosynthesis in simple terms"*

**What happens:**
- Analytics Agent retrieves Alex's profile (avg 86.8%, struggles with essays)
- Tutor Agent searches uploaded PDFs for photosynthesis content
- Tutor crafts a beginner-friendly explanation with analogies + 3 practice questions

---

### Example 2 — Admin Support

1. Select **"🏫 Admin Support"** mode.
2. Ask: *"What are the admission requirements?"*

**What happens:**
- Support Agent searches the policy knowledge base
- Returns specific GPA thresholds, required documents, deadlines, and tips

---

### Example 3 — Progress Report

1. Select **"📊 Progress Report"** mode.
2. Click **"Generate AI Progress Report"**.

**What happens:**
- Analytics Agent calculates trend, ranks strengths/weaknesses, predicts semester outcome
- Tutor Agent creates a 4-week personalised study plan with SMART goals
- Download the full report as a `.txt` file

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Multi-Agent Framework** | CrewAI |
| **LLM Integration** | LangChain + LangChain-OpenAI |
| **Language Models** | GPT-4 (Tutor, Analytics), GPT-3.5-turbo (Support) |
| **Vector Database** | ChromaDB |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **PDF Processing** | PyPDF + LangChain document loaders |
| **Web UI** | Streamlit |
| **Data Storage** | JSON files (student profiles) |
| **Environment** | python-dotenv |

---

## 🔮 Future Enhancements

- [ ] Streaming responses (token-by-token display)
- [ ] Multi-student leaderboard / cohort analytics
- [ ] LTI integration for LMS platforms (Canvas, Moodle)
- [ ] Voice input/output via Whisper + TTS
- [ ] Spaced-repetition flashcard generation from uploaded materials
- [ ] Email progress reports via SendGrid
- [ ] PostgreSQL backend for production-scale student data
- [ ] Agent memory using LangChain `ConversationBufferMemory`
- [ ] Fine-tuned models per subject domain

---

## 🐛 Troubleshooting

**`AuthenticationError`** — Check `OPENAI_API_KEY` in your `.env` file.

**`chromadb` import error** — Run `pip install chromadb>=0.4.22`.

**Slow responses** — GPT-4 can take 20–40 s for multi-step reasoning. Use the spinner as your patience indicator.

**PDF not found in search** — Upload the PDF via the sidebar first; the indexing runs automatically when an API key is present.

---

## 📄 License

MIT — free to use, modify, and distribute.
