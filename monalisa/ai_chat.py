"""
AI Chat Module for generating Mona Lisa's responses
"""
from config import OPENAI_API_KEY, USE_OPENAI, MONA_LISA_SYSTEM_PROMPT

class MonaLisaChat:
    def __init__(self):
        self.conversation_history = []
        if USE_OPENAI:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                print("OpenAI package not installed. Using simple responses.")
                self.client = None
        else:
            self.client = None
    
    def get_response(self, user_message):
        """
        Get response from Mona Lisa
        Falls back to simple rule-based responses if OpenAI is not available
        """
        if USE_OPENAI and self.client:
            return self._get_openai_response(user_message)
        else:
            return self._get_simple_response(user_message)
    
    def _get_openai_response(self, user_message):
        """Get response using OpenAI API"""
        try:
            messages = [
                {"role": "system", "content": MONA_LISA_SYSTEM_PROMPT}
            ] + self.conversation_history + [
                {"role": "user", "content": user_message}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.8
            )
            
            assistant_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_message
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return self._get_simple_response(user_message)
    
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
