"""
AI Chat Module for generating Mona Lisa's responses
Order: OpenAI (if key works) → Ollama (local, free) → rule-based
"""
from config import (
    OPENAI_API_KEY,
    USE_OPENAI,
    MONA_LISA_SYSTEM_PROMPT,
    OLLAMA_URL,
    OLLAMA_MODEL,
)
import requests

class MonaLisaChat:
    def __init__(self):
        self.conversation_history = []
        self.openai_client = None
        if USE_OPENAI:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                print("OpenAI package not installed.")
            except (TypeError, Exception) as e:
                print(f"OpenAI client init failed ({e}).")

    def get_response(self, user_message):
        """
        Get response: try OpenAI → then Ollama → then rule-based.
        """
        # 1) Try OpenAI
        if USE_OPENAI and self.openai_client:
            try:
                return self._get_openai_response(user_message)
            except Exception as e:
                print(f"Error with OpenAI: {e}")
        # 2) Try Ollama (local, free)
        try:
            return self._get_ollama_response(user_message)
        except Exception as e:
            print(f"Ollama not available: {e}")
        # 3) Rule-based fallback
        return self._get_simple_response(user_message)

    def _get_openai_response(self, user_message):
        """Get response using OpenAI API"""
        messages = [
            {"role": "system", "content": MONA_LISA_SYSTEM_PROMPT}
        ] + self.conversation_history + [
            {"role": "user", "content": user_message}
        ]
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.8
        )
        assistant_message = response.choices[0].message.content
        self._append_history(user_message, assistant_message)
        return assistant_message

    def _get_ollama_response(self, user_message):
        """Get response using local Ollama API (free). Tries /api/chat first, then /api/generate."""
        messages = [
            {"role": "system", "content": MONA_LISA_SYSTEM_PROMPT}
        ] + self.conversation_history + [
            {"role": "user", "content": user_message}
        ]
        # Try chat endpoint first (Ollama 0.1.15+)
        chat_url = OLLAMA_URL if "/chat" in OLLAMA_URL else OLLAMA_URL.rstrip("/") + "/api/chat"
        try:
            resp = requests.post(
                chat_url,
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            assistant_message = (data.get("message") or {}).get("content", "").strip()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # Fallback: use /api/generate (older Ollama or different install)
                assistant_message = self._ollama_generate(messages, user_message)
            else:
                raise
        if not assistant_message:
            raise ValueError("Ollama returned empty response")
        self._append_history(user_message, assistant_message)
        return assistant_message

    def _ollama_generate(self, messages, user_message):
        """Use /api/generate when /api/chat is not available (single prompt, no history)."""
        base = OLLAMA_URL.split("/api/")[0] if "/api/" in OLLAMA_URL else OLLAMA_URL.rstrip("/")
        url = base.rstrip("/") + "/api/generate"
        prompt_parts = [MONA_LISA_SYSTEM_PROMPT]
        for m in self.conversation_history[-10:]:  # last 5 exchanges
            role = "Human" if m["role"] == "user" else "Mona Lisa"
            prompt_parts.append(f"{role}: {m['content']}")
        prompt_parts.append(f"Human: {user_message}")
        prompt_parts.append("Mona Lisa:")
        prompt = "\n\n".join(prompt_parts)
        resp = requests.post(
            url,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _append_history(self, user_message, assistant_message):
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def _get_simple_response(self, user_message):
        """Simple rule-based responses when OpenAI is not available"""
        message_lower = user_message.lower()
        
        # Simple keyword-based responses
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Ah, a visitor. What mysteries do you seek to uncover today?"
        elif any(word in message_lower for word in ["art", "painting", "leonardo"]):
            return "Art captures moments that words cannot express. What do you see in my smile?"
        elif any(word in message_lower for word in ["smile", "happy", "mystery"]):
            return "My smile holds many secrets. Perhaps it is for you to discover them."
        elif any(word in message_lower for word in ["how", "why", "what"]):
            return "Questions lead to understanding. What is it you truly wish to know?"
        elif any(word in message_lower for word in ["beautiful", "pretty", "beauty"]):
            return "Beauty is in the eye of the beholder, and in the mystery of the moment."
        else:
            responses = [
                "Interesting. Tell me more of your thoughts.",
                "The world is full of wonders, is it not?",
                "What else stirs your curiosity?",
                "I have observed much in my time. What do you observe?",
                "Every conversation is a journey. Where shall ours lead?"
            ]
            import random
            return random.choice(responses)
