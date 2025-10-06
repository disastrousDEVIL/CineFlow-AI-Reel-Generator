# 🎥 CineFlow: Automated AI Reel Generator

**CineFlow** is an intelligent system that transforms text themes into cinematic short reels using **Google Vertex AI**.
It combines **LLMs (for storytelling)**, **Imagen (for character image generation)**, and **Veo (for video synthesis)** to create complete, ready-to-upload reels automatically.

---

## ⚡ Overview

**Fully Functional** – Works end-to-end from theme to final video.

**Pipeline Includes:**

* 🧠 Story generation using LangChain + GPT-4o
* 🖼️ Character image creation with Imagen
* 🎥 Beat-wise video generation via Veo 3.0
* 🎬 Automatic stitching into a final reel
* ✅ 9:16 portrait format for Reels, TikTok, and Shorts

**Latest Output:**
🎞️ `final_reel.mp4` (46s, 32 MB)
🎭 Theme: *“Overcoming heartbreak and rediscovering purpose”*

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
```

### Set GCP Credentials

**Windows PowerShell**

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\\path\\to\\service-account.json"
```

**macOS/Linux**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Run CineFlow

```bash
python backend/main.py
```

**Output:** `final_reel.mp4` (9:16 portrait, ready to upload)

---

## 💻 Example Commands

**PowerShell (Windows)**

```powershell
# Auto theme end-to-end
python backend\main.py

# Custom theme
$env:THEME="finding hope in a storm"; python backend\main.py

# Rerun fresh
Remove-Item .\story.json -ErrorAction SilentlyContinue; python backend\main.py
```

**Bash (macOS/Linux)**

```bash
# Auto theme end-to-end
python backend/main.py

# Custom theme
THEME="finding hope in a storm" python backend/main.py

# Rerun fresh
rm -f story.json && python backend/main.py
```

---

## 🧩 Extensibility

1. Add multiple characters to a single story
2. Support various visual styles (realistic, anime, cinematic)
3. Integrate AI background music generation
4. Add a web app for easy use
5. Include better error recovery and auto-resume

---

## 📁 Project Structure

```
backend/
 ├── main.py         # Workflow orchestrator
 ├── story_gen.py    # Story generation (LangChain)
 ├── characters.py   # Character image creation (Imagen)
 ├── video.py        # Video generation + stitching
 └── ...
final_reel.mp4       # Output video
story.json           # Generated story data
```

---

## ⚙️ Requirements

* **Python 3.11+**
* **ffmpeg** (for stitching)
* **Google Cloud Project** with Vertex AI enabled
* **Service account JSON** with proper permissions

---

## 🎨 Supported Models

* `veo-3.0-generate-001` *(recommended)*
* `veo-3.0-fast-generate-001` *(faster, cheaper)*
* `imagegeneration@005` *(Imagen model)*

---

## 🧠 Typical Runtime

| Step                | Duration           |
| ------------------- | ------------------ |
| Story Generation    | 5–10 sec           |
| Character Images    | ~1 min each        |
| Video Generation    | 40–80 sec per beat |
| Full Reel (7 beats) | 7–10 mins total    |

---

## 💡 Example Use

```bash
$env:THEME="a detective solving a mystery in noir style"
$env:DURATION="45"
python backend/main.py
```

---

## 🔮 Future Improvements

1. **Add multiple characters** – The system can be expanded to include more than one character in the story, allowing scenes with interactions, emotions, and dialogue between characters.
2. **Change visual style** – Users can choose different art styles such as anime, cartoon, realistic, or cinematic before generating images to match the mood or theme.
3. **Add background music** – Background music or emotional soundtracks can be automatically generated and added to the final video for better storytelling.
4. **Build a web or app interface** – A simple web or mobile interface can be created so users can easily enter a theme and generate their reel without running any code.
5. **Better error recovery** – The system can include automatic retry, progress saving, and resume options so that if an error occurs, it continues from the last successful step instead of restarting.

---

## ❤️ Credits

* **Google Vertex AI** – Imagen & Veo
* **LangChain** – Story and logic orchestration
* **ffmpeg** – Video stitching
* **Developed by Krish Batra**