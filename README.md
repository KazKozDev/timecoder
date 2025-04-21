
<div align="center">
  <img src="https://github.com/user-attachments/assets/056fde13-5c55-41c8-8903-c4628b4ee1a4" alt="timecode_logo">
</div>

## Timecoder

A tool for analyzing YouTube video transcripts, segmenting them by topic, and generating timestamped summaries using AI.

### Features

* Extracts transcripts from YouTube videos via YouTubeTranscriptApi
* Segments transcripts into semantic groups using SentenceTransformer (`all-MiniLM-L6-v2`)
* Annotates topics with KeyBERT or summarization (DistilBART)
* Post-processes segments with a local LLM (Gemma3:12b) for improved punctuation and concise topic titles
* GUI built with Tkinter and ttkbootstrap for user interaction
* Displays results in Markdown format with timestamps
* Logs processing steps and errors
  
![Screenshot](https://github.com/user-attachments/assets/e41ee732-2dd7-4cee-8153-7b43bcb52c2b)

### Requirements

* Python 3.8+
* Dependencies (see `requirements.txt`):
   * regex
   * nltk
   * sentence-transformers
   * transformers
   * keybert
   * torch
   * youtube-transcript-api
   * requests
   * ttkbootstrap
   * markdown
   * tkinterweb
* Ollama with Gemma3:12b model (11GB, requires stable internet and disk space)
* NLTK data: `punkt`, `stopwords`

### Installation

1. Clone the repository:

```
git clone https://github.com/KazKozDev/timecoder.git
cd timecoder
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Install Ollama and pull the model:

```
# Linux/WSL
curl -fsSL https://ollama.com/install.sh | sh
# macOS
brew install ollama
# Windows: Download from https://ollama.com/download
ollama serve
ollama pull gemma3:12b
```

5. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Usage

1. Run the application:

```
python timecoder.py
```

2. Enter a YouTube video URL in the GUI
3. Click "Analyze Transcript" to process
4. View timestamped segments in the output window
5. Copy results with the "Copy All" button
6. Adjust font size via the dropdown (8-16px)

### Notes

* Requires a valid YouTube URL with an available transcript
* Processing time depends on video length and system resources
* All operations run locally for privacy
* Logs are displayed in the GUI and can be reviewed for errors
* Hugging Face authentication is included (replace token in code)

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
