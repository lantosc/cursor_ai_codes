# How to use Ollama with Mona Lisa Chat (free local LLM)

Ollama runs AI models on your PC. No API key or payment needed.

## 1. Install Ollama

- **Windows / Mac / Linux:** Go to **https://ollama.com** and download the installer.
- Run the installer. Ollama will start in the background (you may see an icon in the system tray).

## 2. Check that Ollama is running

Open a terminal (PowerShell, Command Prompt, or your OS terminal) and run:

```bash
ollama list
```

- If you see a list (or “no models” and no error), Ollama is running.
- If you get “command not found” or “connection refused”, install Ollama from step 1 and make sure it’s running.

## 3. Pull a model (download it)

In the **same terminal**, run one of these. The first time will download the model (can take a few minutes):

```bash
ollama pull llama3.2
```

Other small, fast options:

```bash
ollama pull phi3
ollama pull mistral
ollama pull gemma2:2b
```

Use the **exact name** you pulled (e.g. `llama3.2`, `phi3`, `mistral`) in the app.

## 4. Tell Mona Lisa which model to use

- **Default:** The app uses `llama3.2` if you ran `ollama pull llama3.2`.
- **Another model:** In the `monalisa` folder, in your `.env` file, add (or edit):

```env
OLLAMA_MODEL=phi3
```

Use the same name you used with `ollama pull` (e.g. `phi3`, `mistral`, `gemma2:2b`).

## 5. Run the app

From the `monalisa` folder:

```bash
python main.py
```

When OpenAI is not used (no key or quota), the app will use Ollama. If you see “Ollama not available”, check:

- Ollama is running (`ollama list` works).
- You have pulled at least one model (`ollama pull llama3.2` or similar).
- `OLLAMA_MODEL` in `.env` matches a model you pulled (or leave unset to use `llama3.2`).

## Quick reference

| Step            | Command / action                          |
|-----------------|-------------------------------------------|
| Install         | Download from https://ollama.com          |
| Check running   | `ollama list`                             |
| Download model  | `ollama pull llama3.2` (or `phi3`, etc.)  |
| Set model in app| In `.env`: `OLLAMA_MODEL=llama3.2`       |
