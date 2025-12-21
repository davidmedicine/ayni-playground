import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import requests
import streamlit as st

# Optional import for GGUF mode (llama.cpp via python bindings)
try:
    from llama_cpp import Llama  # type: ignore
    LLAMA_CPP_AVAILABLE = True
except Exception:
    LLAMA_CPP_AVAILABLE = False


# ----------------------------
# Config / Types
# ----------------------------

@dataclass
class GenParams:
    temperature: float
    top_p: float
    max_tokens: int


DEFAULT_SYSTEM_ES = (
    "Eres Ayni Guardian, un asistente local y offline-first. "
    "Sé claro, breve y útil. Si no estás seguro, dilo."
)

DEFAULT_SYSTEM_EN = (
    "You are Ayni Guardian, a local offline-first assistant. "
    "Be clear, brief, and helpful. If unsure, say so."
)


# ----------------------------
# Helpers
# ----------------------------

def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {"role": "...", "content": "..."}
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_ES
    if "backend" not in st.session_state:
        st.session_state.backend = "ollama"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "tinyllama"
    if "gguf_path" not in st.session_state:
        st.session_state.gguf_path = ""
    if "gguf_ctx" not in st.session_state:
        st.session_state.gguf_ctx = 2048
    if "gguf_threads" not in st.session_state:
        st.session_state.gguf_threads = max(2, (os.cpu_count() or 4) - 2)
    if "gguf_gpu_layers" not in st.session_state:
        st.session_state.gguf_gpu_layers = 0


@st.cache_resource(show_spinner=False)
def load_gguf_model(model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int):
    if not LLAMA_CPP_AVAILABLE:
        raise RuntimeError("llama-cpp-python is not installed. Use Ollama backend or install llama-cpp-python.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    # llama-cpp-python model loader
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def build_chat_messages(system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history)
    return msgs


# ----------------------------
# Ollama backend
# ----------------------------

def ollama_stream_chat(
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    params: GenParams,
) -> Generator[str, None, None]:
    """
    Stream responses from Ollama local server.
    Ollama API returns newline-delimited JSON objects.
    """
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "num_predict": params.max_tokens,
        },
    }

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                yield data["message"]["content"]
            if data.get("done", False):
                break


# ----------------------------
# GGUF backend (llama.cpp)
# ----------------------------

def gguf_stream_chat(
    llm: "Llama",
    messages: List[Dict[str, str]],
    params: GenParams,
) -> Generator[str, None, None]:
    """
    Stream responses from llama-cpp-python chat completion.
    """
    # Newer llama-cpp-python supports create_chat_completion(stream=True)
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_tokens,
        stream=True,
    )

    for chunk in stream:
        # chunk format can vary slightly across versions
        delta = ""
        try:
            delta = chunk["choices"][0]["delta"].get("content", "")
        except Exception:
            # fallback
            try:
                delta = chunk["choices"][0].get("text", "")
            except Exception:
                delta = ""
        if delta:
            yield delta


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Ayni Guardian Playground", layout="wide")
ensure_session_state()

st.title("Ayni Guardian Playground")
st.caption("Simple local Streamlit UI to test local models (Ollama or GGUF/llama.cpp).")

with st.sidebar:
    st.subheader("Backend")
    backend = st.radio(
        "Choose inference backend",
        options=["ollama", "gguf"],
        index=0 if st.session_state.backend == "ollama" else 1,
        help="Ollama = easiest. GGUF = direct llama.cpp via python bindings.",
    )
    st.session_state.backend = backend

    st.subheader("System prompt")
    st.session_state.system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=120,
    )

    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    max_tokens = st.slider("Max tokens", 16, 512, 192, 8)
    gen_params = GenParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    st.divider()

    if backend == "ollama":
        st.subheader("Ollama")
        host = st.text_input("Ollama host", value="http://127.0.0.1:11434")
        st.session_state.ollama_model = st.text_input("Model name", value=st.session_state.ollama_model)
        st.caption("Tip: run `ollama serve` and `ollama pull <model>` first.")
    else:
        st.subheader("GGUF / llama.cpp")
        if not LLAMA_CPP_AVAILABLE:
            st.error("llama-cpp-python not installed. See README for installation.")
        st.session_state.gguf_path = st.text_input(
            "Path to .gguf model",
            value=st.session_state.gguf_path,
            placeholder="models/tinyllama-q4.gguf",
        )
        st.session_state.gguf_ctx = st.number_input(
            "Context (n_ctx)",
            min_value=512,
            max_value=8192,
            value=st.session_state.gguf_ctx,
            step=256,
        )
        st.session_state.gguf_threads = st.number_input(
            "CPU threads",
            min_value=1,
            max_value=64,
            value=st.session_state.gguf_threads,
            step=1,
        )
        st.session_state.gguf_gpu_layers = st.number_input(
            "GPU layers (0 = CPU only)",
            min_value=0,
            max_value=200,
            value=st.session_state.gguf_gpu_layers,
            step=1,
        )

        st.caption("Put models in ./models/ and keep them out of git (already in .gitignore).")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_prompt = st.chat_input("Type a message…")

if user_prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepare assistant response container
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assembled = ""

        messages = build_chat_messages(st.session_state.system_prompt, st.session_state.messages)

        try:
            if st.session_state.backend == "ollama":
                for token in ollama_stream_chat(
                    host=host,
                    model=st.session_state.ollama_model,
                    messages=messages,
                    params=gen_params,
                ):
                    assembled += token
                    placeholder.markdown(assembled)
            else:
                llm = load_gguf_model(
                    model_path=st.session_state.gguf_path,
                    n_ctx=int(st.session_state.gguf_ctx),
                    n_threads=int(st.session_state.gguf_threads),
                    n_gpu_layers=int(st.session_state.gguf_gpu_layers),
                )
                for token in gguf_stream_chat(llm=llm, messages=messages, params=gen_params):
                    assembled += token
                    placeholder.markdown(assembled)

        except requests.exceptions.ConnectionError:
            placeholder.error("Could not connect to Ollama. Is `ollama serve` running on the host/port?")
        except Exception as e:
            placeholder.error(f"Error: {e}")

    # Append assistant message (even if empty, keep UI consistent)
    st.session_state.messages.append({"role": "assistant", "content": assembled if assembled else "(no output)"})
