"""
Text-to-Speech Module for generating Mona Lisa's voice
"""
from gtts import gTTS
import os
import tempfile
import pygame
import threading

class TextToSpeech:
    def __init__(self, lang='en', slow=False):
        self.lang = lang
        self.slow = slow
        pygame.mixer.init()
        self.is_speaking = False
    
    def generate_audio(self, text):
        """
        Generate audio file from text (without playing)
        Returns audio file path for animation synchronization
        """
        if not text:
            return None
        
        try:
            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            # Generate speech
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(temp_path)
            
            return temp_path
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
    
    def speak(self, text, callback=None):
        """
        Convert text to speech and play it
        Returns audio file path for animation synchronization
        """
        if not text:
            return None
        
        try:
            # Generate audio file
            temp_path = self.generate_audio(text)
            if not temp_path:
                return None
            
            # Play audio in a separate thread
            self.is_speaking = True
            thread = threading.Thread(target=self._play_audio, args=(temp_path, callback))
            thread.daemon = True
            thread.start()
            
            return temp_path
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            self.is_speaking = False
            return None
    
    def play_audio_file(self, audio_path, callback=None):
        """
        Play an existing audio file (no generation). Use when you already have
        the file and want to start playback in sync with animation.
        """
        if not audio_path or not os.path.exists(audio_path):
            if callback:
                callback()
            return
        self.is_speaking = True
        thread = threading.Thread(target=self._play_audio, args=(audio_path, callback))
        thread.daemon = True
        thread.start()
    
    def _play_audio(self, audio_path, callback):
        """Play audio file"""
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            self.is_speaking = False
            
            # Clean up
            try:
                os.unlink(audio_path)
            except:
                pass
            
            if callback:
                callback()
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_speaking = False
    
    def stop(self):
        """Stop current speech"""
        pygame.mixer.music.stop()
        self.is_speaking = False
