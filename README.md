🔬 Research Agent
An AI-powered research assistant that autonomously searches the web, reads sources, and synthesizes professional research reports in real time.

✨ Features

🧠 Autonomous Agent — Plans research, decides what to search, and how many searches to run based on topic complexity
🔍 Focused Research — Optionally guide the agent with a specific angle (e.g. "future applications", "risks", "recent breakthroughs")
📡 Live Activity Feed — Watch the agent think, search, read, and write in real time via SSE streaming
📄 Executive Reports — Generates structured markdown reports with Executive Summary, Key Findings, and Sources
⬇️ Download Reports — Export the final report as a .md file
🌐 Modern UI — Dark theme with emerald/teal accents, glassmorphism, and smooth animations


🛠️ Tech Stack
ComponentTechnologyBackendFastAPILLMLlama 3.3 70b via Groq APIWeb SearchTavily APIStreamingServer-Sent Events (SSE)FrontendVanilla HTML/CSS/JSFontsSpace Grotesk + Inter

🚀 Getting Started
Prerequisites

Python 3.10+
Groq API key (free at console.groq.com)
Tavily API key (free at tavily.com)

Installation

Clone the repository

git clone https://github.com/sarthak742/research-agent-.git
cd research-agent-

Create a virtual environment

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

Install dependencies

pip install -r requirements.txt

Set up environment variables

Create a .env file in the project root:
envGROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

Run the app

python -m uvicorn app:app --reload --port 8000
Open your browser at http://localhost:8000

💡 Usage

Enter a research topic (e.g. "Quantum Computing")
Optionally add a research focus to guide the agent (e.g. "how it will shape the future")
Select Basic (3 searches) or Detailed (7 searches) depth
Click Start Research and watch the agent work live
Read the generated report and download it as .md


🤖 How It Works
User Input
    ↓
Groq LLM creates research plan (dynamic search queries)
    ↓
Tavily searches the web for each aspect
    ↓
Agent reads and extracts key information
    ↓
Groq synthesizes findings into executive report
    ↓
Report streamed live to frontend
The agent dynamically determines how many searches to run based on topic complexity — anywhere from 2 to 8 searches per session.

📁 Project Structure
research-agent/
├── app.py              # FastAPI backend with SSE endpoints
├── agent.py            # Research agent (Groq + Tavily)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore
└── static/
    └── index.html      # Frontend UI

🔑 API Endpoints
EndpointMethodDescription/GETServes the frontend/researchPOSTStarts a research session/stream/{id}GETSSE stream of agent activity/report/{id}GETFetch the final report
