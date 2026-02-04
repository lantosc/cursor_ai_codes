"""
Speech Recognition Module for capturing user input
"""
import speech_recognition as sr
import threading

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready to listen!")
    
    def listen(self, timeout=5, phrase_time_limit=10):
        """
        Listen for speech and return transcribed text
        Returns: (success: bool, text: str or error message)
        """
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            print("Processing speech...")
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return True, text
            except sr.UnknownValueError:
                return False, "Could not understand audio"
            except sr.RequestError as e:
                return False, f"Error with speech recognition service: {e}"
        except sr.WaitTimeoutError:
            return False, "No speech detected"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def listen_continuous(self, callback, stop_event):
        """
        Continuously listen for speech and call callback with results
        """
        self.is_listening = True
        while self.is_listening and not stop_event.is_set():
            success, result = self.listen()
            if success:
                callback(result)
            elif "No speech detected" not in result:
                callback(None)  # Signal error
