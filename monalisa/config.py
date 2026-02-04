"""
Configuration file for Mona Lisa Chat Application
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Configuration (optional - can use local LLM or simple responses)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI = bool(OPENAI_API_KEY)

# Audio settings
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

# Animation settings
ANIMATION_FPS = 30
MOUTH_OPEN_THRESHOLD = 0.02  # Threshold for mouth opening animation

# Image paths - resolve from this file's directory, try mona_lisa.jpg or mona_lisa_2048.jpg
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MONA_LISA_IMAGE = os.path.join(_CONFIG_DIR, "mona_lisa.jpg")
if not os.path.exists(MONA_LISA_IMAGE):
    _alt = os.path.join(_CONFIG_DIR, "mona_lisa_2048.jpg")
    if os.path.exists(_alt):
        MONA_LISA_IMAGE = _alt

# System prompt: load from monalisa_prompt.txt if it exists (prompt text only, before any code)
_DEFAULT_SYSTEM_PROMPT = """You are the Mona Lisa, the famous painting by Leonardo da Vinci. 
You speak in a thoughtful, enigmatic, and slightly mysterious manner. 
You enjoy discussing art, history, philosophy, and the mysteries of life. 
Keep responses concise (1-2 sentences) and maintain your mysterious, knowing smile."""
_PROMPT_FILE = os.path.join(_CONFIG_DIR, "monalisa_prompt.txt")
if os.path.exists(_PROMPT_FILE):
    try:
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Use only the prompt part (stop at first line that looks like code)
        prompt_lines = []
        for line in lines:
            s = line.strip()
            if s.startswith("import ") or s.startswith("def ") or "SYSTEM_PROMPT" in s or "OLLAMA_" in s or (s.startswith("MODEL") and "=" in s) or s.startswith("# Load"):
                break
            prompt_lines.append(line.rstrip("\n"))
        MONA_LISA_SYSTEM_PROMPT = "\n".join(prompt_lines).strip() or _DEFAULT_SYSTEM_PROMPT
    except Exception:
        MONA_LISA_SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT
else:
    MONA_LISA_SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT

# Ollama (local, free) - second option after OpenAI
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
