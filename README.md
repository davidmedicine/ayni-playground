# Ayni Guardian Playground (Streamlit)

A minimal local Streamlit app to test **local LLMs**.

Supports two backends:
- **Ollama** (recommended for ease)
- **GGUF / llama.cpp** via `llama-cpp-python`

## Quick start

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2A) Run with Ollama (easiest)

Install Ollama.

Start server:
```bash
ollama serve
```

Pull a small model:
```bash
ollama pull tinyllama
```

Run Streamlit:
```bash
streamlit run app.py
```

In the sidebar:
- Backend: `ollama`
- Model name: `tinyllama`

### 2B) Run with GGUF (direct llama.cpp)

Download a GGUF model file (quantized) and place it in `./models/`.

Run:
```bash
streamlit run app.py
```

In the sidebar:
- Backend: `gguf`
- Path: `models/<your-model>.gguf`

### Hardware guidance

Start with a small quantized model (GGUF q4). If laptops are weak, prefer:
- Ollama + tiny model, or
- one lab machine serving Ollama on LAN.

## Repo details page

### Ayni Guardian Playground (Español)

Un entorno simple para probar modelos de lenguaje locales en tu propia máquina.

Este prototipo forma parte del programa Ayni Intelligence: construimos herramientas offline-first para comunicación y conocimiento local en los Altos Andes—pensadas para funcionar cuando el internet falla y para respetar la soberanía de datos.

#### Qué puedes hacer aquí
- Probar un modelo local en modo chat (sin APIs externas).
- Cambiar el system prompt y parámetros como temperatura y tokens.
- Comparar dos formas de correr modelos:
  - Ollama (recomendado): más fácil de instalar y usar.
  - GGUF / llama.cpp: cargar un archivo .gguf directamente.

#### Importante
- Este demo no sube tus mensajes a la nube.
- Los modelos no vienen incluidos en este repositorio. Para usar IA local, necesitas:
  - descargar un modelo pequeño (ej. TinyLlama), o
  - conectarte a un servidor local (por ejemplo, una PC del laboratorio en la misma red).

#### Buenas prácticas (fase 1)
- Usa datasets públicos/permitidos o sintéticos.
- Evita datos personales o sensibles.
- Si el modelo no está seguro, es mejor que lo diga claramente.

Objetivo: mantenerlo simple y reproducible para pruebas rápidas, benchmarks y demos en laboratorio.

### Ayni Guardian Playground (English)

A simple environment to test local language models on your own machine.

This prototype is part of Ayni Intelligence: we build offline-first tools for local communication and knowledge in the High Andes—designed to keep working when the internet fails and to respect data sovereignty.

#### What you can do here
- Test a local model in chat mode (no external APIs).
- Edit the system prompt and tune parameters like temperature and max tokens.
- Compare two ways of running models:
  - Ollama (recommended): easiest installation and workflow.
  - GGUF / llama.cpp: load a .gguf file directly.

#### Important
- This demo does not send your messages to the cloud.
- Models are not included in this repository. To use local AI, you’ll need to:
  - download a small model (e.g., TinyLlama), or
  - connect to a local server (for example, a lab PC on the same network).

#### Good practices (Phase 1)
- Use public/permitted or synthetic datasets.
- Avoid personal or sensitive data.
- If the model isn’t sure, it should say so.

Goal: keep it simple and reproducible for quick tests, benchmarks, and lab demos.
# ayni-playground
