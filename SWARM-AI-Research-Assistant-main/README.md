# Swarm AI Research Assistant

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

Swarm AI Research Assistant is a multi-agent AI application that automates the research process. It can break a topic into tasks, collect web and academic information, check important claims, summarize findings, and create a downloadable PDF research report.

The project is built with Python, LangGraph, LangChain, Ollama, Llama 3.2, and Streamlit.

## Project Idea

Manual research usually involves searching different sources, comparing information, verifying facts, and preparing a structured report. This project divides those responsibilities across specialized AI agents instead of using one model for the complete task.

## Research Workflow

```text
User
 |
 v
Coordinator
 |
 +--> Web Researcher
 |
 +--> Academic Researcher
 |
 v
Fact Checker
 |
 v
Summarizer
 |
 v
Report Generator
 |
 v
PDF Report
```

## Features

- Multi-agent AI research workflow.
- Research task planning and coordination.
- Web research.
- Academic paper research.
- Fact checking and claim verification.
- Automated summary generation.
- PDF report creation.
- Streamlit interface.
- Research statistics.
- Source references.
- Downloadable reports.
- Local LLM support through Ollama.
- Sensitive files protected through `.gitignore`.

## AI Agents

### Coordinator Agent

The Coordinator receives the user's topic, understands the research requirement, divides the topic into subtasks, organizes the workflow, and coordinates the remaining agents.

### Web Researcher Agent

The Web Researcher searches online sources and collects titles, URLs, snippets, and supporting information related to the topic.

### Academic Researcher Agent

The Academic Researcher focuses on research papers and academic sources. It collects paper title, authors, publication details, summary, and paper URL.

### Fact Checker Agent

The Fact Checker reviews collected information, checks important claims, compares evidence, identifies weak or unreliable statements, and produces verified findings.

### Summarizer Agent

The Summarizer converts verified information into a structured research summary. Sections can include introduction, key findings, applications, benefits, limitations, future scope, and conclusion.

### Report Generator Agent

The Report Generator creates the final PDF report containing topic, summary, key findings, references, web sources, and academic sources. Reports are stored in the `reports/` folder.

## Technologies

| Technology | Purpose |
| ---------- | ------- |
| Python | Core language |
| LangGraph | Agent workflow orchestration |
| LangChain | LLM application framework |
| Ollama | Local model execution |
| Llama 3.2 | Local language model |
| Streamlit | Web interface |
| DuckDuckGo Search | Web research |
| arXiv | Academic research |
| ReportLab | PDF generation |
| python-dotenv | Environment variable handling |
| Git and GitHub | Version control |

## Project Files

```text
SWARM-AI-Research-Assistant/
|-- agents/
|   |-- academic_researcher.py
|   |-- coordinator.py
|   |-- fact_checker.py
|   |-- report_generator.py
|   |-- summarizer.py
|   `-- web_researcher.py
|-- tools/
|-- workflow/
|   |-- graph.py
|   `-- state.py
|-- reports/
|-- app.py
|-- config.py
|-- requirements.txt
|-- test_research.py
|-- .gitignore
`-- README.md
```

## How the Program Works

The user enters a research topic in the Streamlit app. The Coordinator plans the research tasks. The Web Researcher gathers online information, and the Academic Researcher searches academic papers. The collected data goes to the Fact Checker for verification. The Summarizer creates a structured explanation from verified results, and the Report Generator prepares a PDF report with findings and references.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/swarm-ai-research-assistant.git
cd swarm-ai-research-assistant
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Ollama Setup

This project uses Ollama for running the language model locally. Install Ollama, then pull the model:

```bash
ollama pull llama3.2
```

Check that the model is available:

```bash
ollama list
```

## Environment Variables

Create a `.env` file in the project root if the configuration requires API keys.

Example:

```env
OPENAI_API_KEY=your_api_key_here
```

Do not upload `.env`, API keys, passwords, or tokens to GitHub. The `.gitignore` file is configured to help prevent sensitive files from being committed. If the project runs fully with local Ollama, an OpenAI API key may not be required.

## Running the Project

Streamlit web app:

```powershell
streamlit run app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

Terminal workflow:

```powershell
python test_research.py
```

## Example Topics

```text
Artificial Intelligence in Healthcare
Artificial Intelligence in Cybersecurity
Impact of Machine Learning on Education
Applications of Generative AI
Renewable Energy Technologies
```

## Application Output

The app provides:

- Number of research tasks.
- Number of web sources.
- Number of academic papers.
- Structured research summary.
- Fact-check results.
- Web source links.
- Academic paper details.
- Downloadable PDF report.

## Generated Reports

Reports are saved in:

```text
reports/
```

Example:

```text
reports/
`-- Artificial_Intelligence_in_Healthcare_20260804_163210.pdf
```

The report contains the research topic, summary, key findings, conclusion, references, web sources, and academic sources.

## Multi-Agent Architecture

The system separates the research process into agent roles. This makes the workflow easier to organize, improves source comparison, supports fact verification, and allows new specialized agents to be added later.

## Security Notes

The project uses `.gitignore` to exclude sensitive or unnecessary files such as:

```text
.env
venv/
__pycache__/
*.pdf
.vscode/
.idea/
```

## Future Scope

- Show real-time agent execution status.
- Add more specialized research agents.
- Support more academic databases.
- Add more web search providers.
- Improve PDF formatting.
- Add research visualizations.
- Save research history.
- Export multiple report formats.
- Improve source ranking.
- Score source reliability automatically.
- Add follow-up research questions.
- Add user authentication.
- Deploy to the cloud.
- Execute agents in parallel.
- Add citation management.

## Learning Outcomes

This project covers multi-agent AI systems, swarm AI concepts, agent orchestration, LangGraph workflows, LangChain, local LLMs with Ollama, prompt engineering, web and academic research automation, fact checking, state management, PDF generation, Streamlit app development, environment variable handling, and Git/GitHub usage.

## Acknowledgement

This project was developed as an internship project to explore multi-agent AI, swarm AI, LLM orchestration, and automated research systems.
