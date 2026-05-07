# AI Resume Analyzer

An AI-powered full-stack web application that parses resumes and delivers deep analysis across ATS compatibility, content quality, format structure, and skill gaps — with live job recommendations matched to the candidate's profile.

---

## Features

- **Resume Parsing** — Supports PDF, DOCX, and image formats (JPG, PNG, WEBP) with OCR for scanned documents
- **ATS Scoring** — Keyword match rate, missing keywords, ATS-friendly section detection
- **Content Quality Analysis** — Bullet point strength, summary score, impact scoring with weak/strong bullet identification
- **Format & Structure Review** — Section completeness, consistency, and structural issue detection
- **Skill Gap Analysis** — Matches skills against inferred target role, flags critical gaps and outdated skills
- **Live Job Recommendations** — Surfaces 50+ live job listings via JSearch API matched to inferred role and skills
- **Neo4j Graph Storage** — Stores parsed resume data as a knowledge graph with 8+ node types and 10+ relationship types

---

## Tech Stack

**Frontend**
- React 18 + Vite
- GSAP + ScrollTrigger (animations)
- CSS custom properties (dark theme)

**Backend**
- Node.js + Express
- Python 3 (spawned as child process)
- Multer (file uploads)

**AI / ML**
- OpenAI GPT API
- LangGraph (multi-node analysis pipeline)
- PyPDF2 + python-docx + Tesseract OCR (document parsing)

**Database**
- Neo4j (graph database)

**APIs**
- JSearch via RapidAPI (live job listings)

---

## Project Structure

```
AI-Resume/
├── backend/
│   ├── resume_parser/          # Python parsing + LLM extraction
│   │   ├── config.py
│   │   ├── schemas.py          # Pydantic models
│   │   ├── pipeline.py
│   │   ├── llm_extract.py
│   │   ├── neo4j_store.py
│   │   └── parsers/
│   │       └── extract.py
│   ├── src/
│   │   └── controllers/
│   │       └── uploadResume.js
│   ├── process_url_with_python.py  # Entry point for Node → Python bridge
│   ├── requirements.txt
│   ├── server.js
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   └── package.json
└── main.py                     # CLI tool for local testing
```

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.9+
- Tesseract OCR installed on your system
- Neo4j instance (local or cloud via Neo4j Aura)

**Install Tesseract:**
```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

---

### Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install
```

Create `backend/.env`:
```env
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
PYTHON_BIN=python   # or full path to venv: .venv/Scripts/python.exe
```

Start the backend:
```bash
node server.js
```

Server runs on `http://localhost:3000`

---

### Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env`:
```env
VITE_API_BASE=http://localhost:3000/api
VITE_RAPIDAPI_KEY=your_rapidapi_key
```

Start the frontend:
```bash
npm run dev
```

App runs on `http://localhost:5173`

---

### CLI Usage (local testing)

You can test the full pipeline directly without the frontend:

```bash
cd backend
python main.py
# Enter path to resume file: /path/to/resume.pdf
```

---

## How It Works

```
User uploads resume (PDF / DOCX / Image)
        ↓
Express (multer) — saves to temp file
        ↓
Node.js spawns Python process with temp file path
        ↓
resume_parser/pipeline.py — extracts text
        ↓
llm_extract.py — OpenAI GPT parses structured JSON
        ↓
LangGraph pipeline runs 4 analysis nodes in sequence:
  ├── ATS Check
  ├── Content Quality
  ├── Format & Structure
  └── Skill Gap Analysis
        ↓
neo4j_store.py — stores graph in Neo4j
        ↓
JSON returned to Node.js → sent to React frontend
        ↓
JSearch API — fetches live job matches
```

---

## Environment Variables

| Variable | Location | Description |
|---|---|---|
| `OPENAI_API_KEY` | backend/.env | OpenAI API key |
| `NEO4J_URI` | backend/.env | Neo4j connection URI |
| `NEO4J_USERNAME` | backend/.env | Neo4j username |
| `NEO4J_PASSWORD` | backend/.env | Neo4j password |
| `PYTHON_BIN` | backend/.env | Path to Python executable |
| `VITE_API_BASE` | frontend/.env | Backend API base URL |
| `VITE_RAPIDAPI_KEY` | frontend/.env | RapidAPI key for JSearch |

---

## Deployment

The app is deployed on **Render**:

- **Backend** — Docker-based Web Service (`backend/Dockerfile`)
- **Frontend** — Static Site (`dist/` publish directory)

Set all environment variables in Render's dashboard under each service's Environment tab.

---

## Resume Bullet Points

```
• Built a full-stack AI Resume Analyzer (Node.js + Python + React) with 4 LLM-powered
  modules — ATS scoring, content quality, format review, and skill gap detection —
  using OpenAI API and LangGraph.

• Integrated JSearch RapidAPI to surface 50+ live job matches per resume based on
  inferred role and skills; modeled a Neo4j graph database with 8+ node types to
  store and query candidate profiles.

• Implemented resume parsing for PDF, DOCX, and image formats with keyword extraction
  and ATS compatibility scoring to improve job-role alignment.
```

---

## License

MIT
