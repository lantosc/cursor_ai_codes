"""
Main application for Mona Lisa Chat
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import os

from speech_recognition_module import SpeechRecognizer
from ai_chat import MonaLisaChat
from text_to_speech import TextToSpeech
from face_animation import FaceAnimator
from advanced_face_animation import AdvancedFaceAnimator
from config import MONA_LISA_IMAGE

class MonaLisaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat with Mona Lisa")
        self.root.geometry("900x700")
        
        # Initialize components
        self.speech_recognizer = SpeechRecognizer()
        self.chat_bot = MonaLisaChat()
        self.tts = TextToSpeech()
        
        # Try to load advanced face animator (fallback to basic if needed)
        self.animator = None
        self.use_advanced = True  # Set to False to use basic animator
        
        if os.path.exists(MONA_LISA_IMAGE):
            try:
                print(f"Loading advanced face animator with image: {MONA_LISA_IMAGE}")
                if self.use_advanced:
                    try:
                        self.animator = AdvancedFaceAnimator(MONA_LISA_IMAGE)
                        print("Advanced face animator initialized successfully!")
                    except Exception as e:
                        print(f"Advanced animator failed: {e}")
                        print("Falling back to basic animator...")
                        self.animator = FaceAnimator(MONA_LISA_IMAGE)
                        self.use_advanced = False
                else:
                    self.animator = FaceAnimator(MONA_LISA_IMAGE)
                    self.use_advanced = False
                
                if hasattr(self.animator, 'face_landmarks') and self.animator.face_landmarks:
                    print("Face animator initialized successfully!")
                elif not hasattr(self.animator, 'face_landmarks'):
                    print("Face animator initialized (advanced mode)")
            except Exception as e:
                print(f"Could not initialize animator: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showwarning("Animation Warning", 
                    f"Could not load face animation: {str(e)}\n\nAnimation features will be disabled.")
        
        # GUI Setup
        self.setup_gui()
        
        # Animation thread
        self.animation_thread = None
        self.stop_animation_event = threading.Event()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Mona Lisa Image Panel
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = ttk.Label(image_frame, text="Mona Lisa")
        self.image_label.pack()
        
        # Display initial image
        self.update_image_display()
        
        # Chat Panel
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat history
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            width=50, 
            height=20,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=1)
        
        self.send_button = ttk.Button(button_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=2)
        
        self.listen_button = ttk.Button(button_frame, text="🎤 Listen", command=self.start_listening)
        self.listen_button.pack(side=tk.LEFT, padx=2)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initial message
        self.add_message("Mona Lisa", "Hello. What brings you here today?")
    
    def update_image_display(self):
        """Update the Mona Lisa image display"""
        if self.animator:
            frame = self.animator.get_current_frame()
            # Resize for display
            display_height = 400
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            frame_resized = cv2.resize(frame, (display_width, display_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk  # Keep a reference
        else:
            # Display placeholder
            placeholder = Image.new('RGB', (300, 400), color='gray')
            img_tk = ImageTk.PhotoImage(image=placeholder)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
    
    def add_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self):
        """Send text message"""
        message = self.input_entry.get().strip()
        if not message:
            return
        
        self.input_entry.delete(0, tk.END)
        self.add_message("You", message)
        self.process_response(message)
    
    def start_listening(self):
        """Start listening for speech"""
        self.listen_button.config(state=tk.DISABLED)
        self.status_label.config(text="Listening...")
        
        def listen_thread():
            success, result = self.speech_recognizer.listen()
            self.root.after(0, self.handle_speech_result, success, result)
        
        thread = threading.Thread(target=listen_thread)
        thread.daemon = True
        thread.start()
    
    def handle_speech_result(self, success, result):
        """Handle speech recognition result"""
        self.listen_button.config(state=tk.NORMAL)
        self.status_label.config(text="Ready")
        
        if success:
            self.add_message("You", result)
            self.process_response(result)
        else:
            self.status_label.config(text=f"Error: {result}")
    
    def process_response(self, user_message):
        """Process user message and get response"""
        self.status_label.config(text="Mona Lisa is thinking...")
        
        def response_thread():
            # Get AI response
            response = self.chat_bot.get_response(user_message)
            
            # Update GUI
            self.root.after(0, self.add_message, "Mona Lisa", response)
            self.root.after(0, self.status_label.config, {"text": "Mona Lisa is speaking..."})
            
            # Generate speech and get audio path (once)
            audio_path = None
            try:
                audio_path = self.tts.generate_audio(response)
                if audio_path:
                    print(f"Generated audio file: {audio_path}")
            except Exception as e:
                print(f"Error generating audio: {e}")
            
            estimated_duration = max(2.0, min(5.0, len(response.split()) * 0.5))
            
            # Advanced path: pre-analyze audio in this thread so animation and playback can start together
            if (self.animator and self.use_advanced and audio_path and
                    hasattr(self.animator, 'animate_with_audio') and
                    getattr(self.animator, 'audio_analyzer', None)):
                try:
                    viseme_sequence = self.animator.audio_analyzer.analyze_audio(audio_path)
                    if viseme_sequence:
                        duration = viseme_sequence[-1][0]
                        # Start animation and playback together (no analysis delay in animation thread)
                        self.root.after(0, self._start_animation_and_playback, audio_path, duration, viseme_sequence)
                    else:
                        self.root.after(0, self.start_face_animation, estimated_duration)
                        self.tts.play_audio_file(audio_path, callback=lambda: self.root.after(0, lambda: self.status_label.config(text="Ready")))
                except Exception as e:
                    print(f"Pre-analyze failed: {e}, using basic flow")
                    self.root.after(0, self.start_face_animation, estimated_duration)
                    if audio_path:
                        self.tts.play_audio_file(audio_path, callback=lambda: self.root.after(0, lambda: self.status_label.config(text="Ready")))
                    else:
                        self.tts.speak(response, callback=lambda: self.root.after(0, lambda: self.status_label.config(text="Ready")))
            elif self.animator:
                self.root.after(0, self.start_face_animation, estimated_duration)
                self.tts.speak(response, callback=lambda: self.root.after(0, lambda: self.status_label.config(text="Ready")))
            else:
                self.tts.speak(response, callback=lambda: self.root.after(0, lambda: self.status_label.config(text="Ready")))
        
        thread = threading.Thread(target=response_thread)
        thread.daemon = True
        thread.start()
    
    def start_face_animation(self, duration=3.0):
        """Start basic face animation"""
        if self.animator and not self.animator.is_animating:
            def update_display():
                """Callback to update display from animation thread"""
                self.root.after(0, self.update_image_display)
            
            def animate():
                # Pass update callback to animation
                if hasattr(self.animator, 'animate_mouth'):
                    self.animator.animate_mouth(duration=duration, update_callback=update_display)
                # Final update after animation
                self.root.after(0, self.update_image_display)
            
            self.animation_thread = threading.Thread(target=animate)
            self.animation_thread.daemon = True
            self.animation_thread.start()
            
            # Also update display periodically as backup
            self.update_animation_display()
    
    def _start_animation_and_playback(self, audio_path, duration, viseme_sequence):
        """Start animation (with pre-computed visemes) and audio playback at the same time."""
        if not self.animator or self.animator.is_animating:
            return
        
        def update_display():
            try:
                self.root.after(0, self.update_image_display)
            except Exception:
                pass
        
        def animate():
            try:
                self.animator.animate_with_audio(
                    audio_path, duration=duration, update_callback=update_display,
                    pre_viseme_sequence=viseme_sequence
                )
                self.root.after(0, self.update_image_display)
            except Exception as e:
                print(f"Animation error: {e}")
                if hasattr(self.animator, 'animate_mouth'):
                    self.animator.animate_mouth(duration=duration, update_callback=update_display)
        
        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()
        self.update_animation_display()
        
        # Start playback at the same time (no analysis delay)
        self.tts.play_audio_file(
            audio_path,
            callback=lambda: self.root.after(0, lambda: self.status_label.config(text="Ready"))
        )
    
    def start_advanced_animation(self, audio_path, duration=3.0):
        """Start advanced audio-synchronized animation (analyzes audio in thread; use _start_animation_and_playback for sync)"""
        if not self.animator:
            print("No animator available")
            return
        
        if self.animator.is_animating:
            print("Animation already in progress")
            return
        
        print(f"Starting advanced animation with audio: {audio_path}")
        
        def update_display():
            try:
                self.root.after(0, self.update_image_display)
            except Exception as e:
                print(f"Error updating display: {e}")
        
        def animate():
            try:
                if hasattr(self.animator, 'animate_with_audio'):
                    self.animator.animate_with_audio(audio_path, duration=duration, update_callback=update_display)
                else:
                    if hasattr(self.animator, 'animate_mouth'):
                        self.animator.animate_mouth(duration=duration, update_callback=update_display)
                self.root.after(0, self.update_image_display)
            except Exception as e:
                print(f"Error in animation thread: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if hasattr(self.animator, 'animate_mouth'):
                        self.animator.animate_mouth(duration=duration, update_callback=update_display)
                except Exception as e2:
                    print(f"Basic animation also failed: {e2}")
        
        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()
        self.update_animation_display()
    
    def update_animation_display(self):
        """Update image display during animation"""
        if self.animator:
            self.update_image_display()
            if self.animator.is_animating:
                # Schedule next update (10 FPS = ~100ms) as backup
                self.root.after(100, self.update_animation_display)

def main():
    # Check if Mona Lisa image exists
    if not os.path.exists(MONA_LISA_IMAGE):
        print(f"Warning: {MONA_LISA_IMAGE} not found.")
        print("Please download a Mona Lisa image and save it as 'mona_lisa.jpg' in the monalisa folder.")
        print("You can continue without animation, but face animation will be disabled.")
    
    root = tk.Tk()
    app = MonaLisaApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
