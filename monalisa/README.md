# Chat with Mona Lisa

An interactive application that allows you to chat with the Mona Lisa painting using voice or text, with animated facial movements synchronized to speech.

## Features

- 🎤 **Voice Input**: Speak to Mona Lisa using your microphone
- 💬 **Text Chat**: Type messages to have conversations
- 🤖 **AI Responses**: Intelligent responses from Mona Lisa (using OpenAI GPT or simple rule-based)
- 🔊 **Text-to-Speech**: Mona Lisa responds with voice
- 🎭 **Facial Animation**: Mouth movements synchronized with speech
- 🎨 **Beautiful GUI**: Clean, intuitive interface

## Requirements

- Python 3.8 or higher
- A microphone (for voice input)
- Internet connection (for speech recognition and text-to-speech)
- Optional: OpenAI API key (for better AI responses)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. For PyAudio on Windows, if you encounter issues, try:
```bash
pip install pipwin
pipwin install pyaudio
```

3. Download a Mona Lisa image and save it as `mona_lisa.jpg` in the `monalisa` folder.
   - You can download one from: https://en.wikipedia.org/wiki/Mona_Lisa#/media/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg

4. (Optional) For better AI responses, create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the application:
```bash
python main.py
```

### Using the Application

1. **Text Chat**: Type your message in the text box and press Enter or click "Send"
2. **Voice Chat**: Click the "🎤 Listen" button and speak your message
3. Watch Mona Lisa respond with animated facial movements and voice!

### Controls

- **Send Button**: Send typed messages
- **🎤 Listen Button**: Activate voice input
- **Enter Key**: Quick send for typed messages

## Project Structure

```
monalisa/
├── main.py                      # Main application and GUI
├── speech_recognition_module.py # Speech-to-text functionality
├── ai_chat.py                   # AI conversation handler
├── text_to_speech.py            # Text-to-speech functionality
├── face_animation.py            # Facial animation module
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── mona_lisa.jpg                # Mona Lisa image (user provided)
└── .env                         # Environment variables (optional)
```

## How It Works

1. **Speech Recognition**: Uses Google Speech Recognition API to convert your voice to text
2. **AI Chat**: Processes your message and generates a response (OpenAI GPT or rule-based)
3. **Text-to-Speech**: Converts Mona Lisa's response to audio using Google Text-to-Speech
4. **Face Animation**: Uses MediaPipe to detect facial landmarks and animates the mouth based on audio

## Troubleshooting

### Microphone Issues
- Make sure your microphone is connected and enabled
- Check microphone permissions in your system settings
- Try adjusting microphone input levels

### Animation Not Working
- Ensure `mona_lisa.jpg` exists in the monalisa folder
- The image should clearly show a face (MediaPipe needs to detect facial landmarks)
- Try a higher resolution image

### Speech Recognition Errors
- Check your internet connection (uses Google Speech Recognition API)
- Try speaking more clearly or in a quieter environment
- Increase the timeout in `speech_recognition_module.py` if needed

### OpenAI API Errors
- If you get API errors, the app will fall back to simple rule-based responses
- Check your API key is correct in the `.env` file
- Ensure you have API credits available

## Future Enhancements

- [ ] More sophisticated lip-sync using audio analysis
- [ ] Better facial expression animations
- [ ] Support for multiple languages
- [ ] Save conversation history
- [ ] More advanced animation techniques (e.g., Wav2Lip integration)

## License

This project is for educational purposes.

## Credits

- Uses Google Speech Recognition API
- Uses Google Text-to-Speech (gTTS)
- Uses MediaPipe for facial landmark detection
- OpenAI API (optional) for AI responses
