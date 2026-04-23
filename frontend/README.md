# AI Resume Analyzer — Frontend

This is a minimal React + Vite frontend for the AI Resume Analyzer.

Quick start:

1. Install dependencies

```bash
cd frontend
npm install
```

2. Run the dev server (default: http://localhost:5173)

```bash
npm run dev
```

3. Ensure the backend is running (default: http://localhost:3000). Upload a resume from the UI — the app will POST to `/api/upload-resume` and display the LLM analysis sections (ATS check, Content Quality, Format & Structure, Skill Gap) as plain text.

Notes:
- The app intentionally hides parsed resume JSON and only shows the four analysis sections.
- If your backend uses a different host/port, set `VITE_API_BASE` in an `.env` file before running, e.g. `VITE_API_BASE=http://localhost:3000/api`.
