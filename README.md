# ayni-playground — Local AI Evaluation (Research Preview)

A minimal Streamlit workspace for testing **local/edge LLMs** with an **offline-first** mindset.

This repo is part of **Qori Labs** (Public Interest Technology Lab): we prototype “Sovereign Layers” for territories with unreliable connectivity—where **models run locally**, data stays local, and governance can be enforced.

**Status:** Research Preview (v0.1)  
**Privacy:** No cloud calls by default. Runs on your machine or a local LAN host.

---

## What this is for

Use this app to:
- Run quick local chat tests against a small model (no external APIs).
- Compare two execution backends:
  - **Ollama** (easiest workflow)
  - **GGUF / llama.cpp** via `llama-cpp-python` (direct model file)
- Tune evaluation parameters (system prompt, temperature, max tokens).
- Keep experiments **reproducible** and aligned with data sovereignty constraints.

---

## Quick start

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
2A) Run with Ollama (recommended)
Install Ollama.

Start server:

bash
Copy code
ollama serve
Pull a small model:

bash
Copy code
ollama pull tinyllama
Run Streamlit:

bash
Copy code
streamlit run app.py
In the sidebar:

Backend: ollama

Model: tinyllama

2B) Run with GGUF (direct llama.cpp)
Place a quantized GGUF file in ./models/.

Run:

bash
Copy code
streamlit run app.py
In the sidebar:

Backend: gguf

Path: models/<your-model>.gguf

Hardware guidance (practical)
Start small:

GGUF q4 quantization is usually the best default.

If a laptop struggles, use either:

Ollama with a tiny model, or

one local “lab” machine serving Ollama on the LAN.

Data + safety (research discipline)
This repo is meant to support Qori Labs’ standards:

Use synthetic, public, or permitted datasets only.

Avoid personal or sensitive data.

Prefer “no PII by design”: don’t paste private identifiers into prompts.

If the model is uncertain, it should say so.

This is evaluation tooling, not a production safety system.

Roadmap (near-term)
Add simple test suites (prompt packs) for repeatable benchmarks

Add “local RAG” mode using a small local knowledge base

Export results summaries for research notes / annexes

License
MIT (unless otherwise stated per component).
