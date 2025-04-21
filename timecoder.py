import sys
import re
import nltk
import json
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from huggingface_hub import login
from keybert import KeyBERT
import torch
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import threading
from queue import Queue, Empty
from markdown import markdown
from tkinterweb.htmlwidgets import HtmlFrame

# Set up logging with a custom handler to capture logs for GUI
log_queue = Queue()
class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

# Clear existing handlers to prevent duplicates
logging.getLogger().handlers = []
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()
queue_handler = QueueHandler()
queue_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(queue_handler)

# Authenticate with Hugging Face
login(token="Insert Hugging Face Token here")
logging.info("Authenticated with Hugging Face")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize KeyBERT with the same model as grouping
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=model)

def get_video_id(youtube_url: str) -> str:
    youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_regex, youtube_url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL or no video ID found.")

def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def clean_transcript(text: str) -> str:
    text = re.sub(r'\[.*?\]', '', text)
    fillers = r'\b(uh|um|you know|like|right,okay,so,well)\b'
    text = re.sub(fillers, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_topic_annotation(text: str) -> str:
    if not text.strip():
        return "Topic annotation unavailable (empty text)"
    
    cleaned_text = clean_transcript(text)
    if not cleaned_text:
        return "Topic annotation unavailable (cleaned text empty)"
    
    try:
        logging.info("Extracting keywords with KeyBERT")
        keywords = kw_model.extract_keywords(
            cleaned_text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=5,
            diversity=0.7
        )
        keywords.sort(key=lambda x: x[1], reverse=True)
        top_keywords = keywords[:3]
        if top_keywords:
            return ", ".join([kw[0] for kw in top_keywords])
    except Exception as ke:
        logging.warning(f"Keyphrase extraction failed: {ke}")
    
    try:
        logging.info("Attempting to use summarization model")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
        max_input_length = 512
        truncated_text = cleaned_text[:max_input_length]
        summary = summarizer(truncated_text, max_length=10, min_length=3, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logging.warning(f"Summarization failed: {e}")
        
    try:
        sentences = nltk.sent_tokenize(cleaned_text)
        if sentences:
            first_sentence = sentences[0]
            return first_sentence[:50] + ("..." if len(first_sentence) > 50 else "")
        return "Topic annotation unavailable"
    except:
        return "Topic annotation unavailable"

def preprocess_segments(segments: list) -> list:
    merged_segments = []
    current_text = ""
    current_start = None
    current_duration = 0
    min_segment_length = 5
    
    for text, start, duration in segments:
        text_clean = clean_transcript(text)
        if len(text_clean) < min_segment_length and current_text:
            current_text += " " + text_clean
            current_duration += duration
        else:
            if current_text:
                merged_segments.append((current_text, current_start, current_duration))
            current_text = text_clean
            current_start = start
            current_duration = duration
    
    if current_text:
        merged_segments.append((current_text, current_start, current_duration))
    
    return merged_segments

def get_transcript(youtube_url: str) -> list:
    try:
        video_id = get_video_id(youtube_url)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        auto_transcript = None
        any_transcript = None
        
        for transcript in transcript_list:
            if transcript.is_generated:
                auto_transcript = transcript
                break
            if not any_transcript:
                any_transcript = transcript
        
        chosen_transcript = auto_transcript or any_transcript
        if not chosen_transcript:
            raise ValueError("No transcript available for this video.")
        
        transcript_data = chosen_transcript.fetch()
        if not transcript_data:
            raise ValueError("Transcript data is empty.")
        
        logging.info(f"Fetched {len(transcript_data)} transcript segments")
        
        segments = [(item['text'], item['start'], item['duration']) for item in transcript_data if item['text'].strip()]
        if not segments:
            raise ValueError("No valid transcript segments found.")
        
        segments = preprocess_segments(segments)
        logging.info(f"After preprocessing, {len(segments)} segments remain")
        
        chunk_size = 8
        chunks = []
        for i in range(0, len(segments), chunk_size):
            chunk_text = ' '.join([s[0] for s in segments[i:i+chunk_size]])
            chunks.append(chunk_text)
        
        logging.info(f"Created {len(chunks)} chunks for semantic analysis")
        
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        
        semantic_groups = []
        current_group = [0]
        similarity_threshold = 0.35
        min_chunks_per_group = 2
        max_chunks_per_group = 8
        
        logging.info("Grouping chunks by semantic content")
        for i in range(1, len(chunks)):
            group_embedding = torch.mean(chunk_embeddings[current_group], dim=0)
            similarity = util.cos_sim(group_embedding, chunk_embeddings[i]).item()
            
            if (similarity >= similarity_threshold and len(current_group) < max_chunks_per_group) or len(current_group) < min_chunks_per_group:
                current_group.append(i)
            else:
                semantic_groups.append(current_group)
                current_group = [i]
        
        if current_group:
            semantic_groups.append(current_group)
        
        logging.info(f"Created {len(semantic_groups)} semantic groups")
        
        final_groups = []
        for group_indices in semantic_groups:
            segment_indices = []
            for chunk_idx in group_indices:
                start_seg_idx = chunk_idx * chunk_size
                end_seg_idx = min(start_seg_idx + chunk_size, len(segments))
                segment_indices.extend(list(range(start_seg_idx, end_seg_idx)))
            
            segment_indices = sorted(list(set(segment_indices)))
            
            if segment_indices:
                start_time = segments[segment_indices[0]][1]
                last_segment = segments[segment_indices[-1]]
                end_time = last_segment[1] + last_segment[2]
                group_segments = [segments[idx][0] for idx in segment_indices]
                group_text = ' '.join(group_segments)
                topic = get_topic_annotation(group_text)
                final_groups.append((start_time, end_time, group_text, topic))
                logging.info(f"Created semantic group: {format_time(start_time)}-{format_time(end_time)}, {len(segment_indices)} segments")
        
        if len(final_groups) <= 1:
            logging.info("Only one group created, trying with more aggressive segmentation")
            num_groups = 5
            segments_per_group = len(segments) // num_groups
            forced_groups = []
            
            for i in range(num_groups):
                start_idx = i * segments_per_group
                end_idx = start_idx + segments_per_group if i < num_groups - 1 else len(segments)
                start_time = segments[start_idx][1]
                last_segment = segments[end_idx - 1]
                end_time = last_segment[1] + last_segment[2]
                group_segments = [segments[idx][0] for idx in range(start_idx, end_idx)]
                group_text = ' '.join(group_segments)
                topic = get_topic_annotation(group_text)
                forced_groups.append((start_time, end_time, group_text, topic))
                logging.info(f"Created forced group: {format_time(start_time)}-{format_time(end_time)}, {end_idx - start_idx} segments")
            
            final_groups = forced_groups
        
        return final_groups
        
    except Exception as e:
        logging.error(f"Error in transcript analysis: {str(e)}")
        raise

def clean_topic(topic: str) -> str:
    """Clean the topic string to ensure proper Markdown formatting."""
    # Remove extra asterisks, spaces, and other characters that might break Markdown
    topic = topic.strip()
    topic = re.sub(r'[\*]+', '', topic)  # Remove any asterisks
    topic = re.sub(r'\s+', ' ', topic)   # Normalize spaces
    return topic

def post_process_with_gemma(transcript_groups):
    logging.info("Post-processing transcript segments with gemma3:12b")
    improved_segments = []
    processed_segments = set()  # Track processed segments to avoid duplicates
    
    for start, end, text, topic in transcript_groups:
        start_str = format_time(start)
        end_str = format_time(end)
        segment_key = f"{start_str}-{end_str}"
        
        if segment_key in processed_segments:
            logging.info(f"Skipping already processed segment {segment_key}")
            continue
        
        try:
            prompt = f"""Please improve only the following transcript segment:
1. Add proper punctuation and capitalization to the text
2. Create a concise, descriptive topic title (3-5 words)

Original segment:
[{start_str} - {end_str}] {topic}
{text}

Improved segment (do not include timestamp in response, just topic and text):
"""
            
            response = requests.post('http://localhost:11434/api/generate', 
                                    json={
                                        "model": "gemma3:12b",
                                        "prompt": prompt,
                                        "stream": False
                                    })
            
            if response.status_code == 200:
                result = response.json()
                improved_text = result.get('response', '')
                
                if improved_text:
                    parts = improved_text.split('\n', 1)
                    if len(parts) >= 2:
                        improved_topic = clean_topic(parts[0].strip())
                        improved_content = parts[1].strip()
                        logging.info(f"Successfully processed segment {start_str}-{end_str} with topic: {improved_topic}")
                    else:
                        logging.warning(f"Unexpected format from gemma3:12b for segment {start_str}-{end_str}")
                        improved_topic = clean_topic(topic)
                        improved_content = improved_text.strip()
                    
                    improved_segments.append((start_str, end_str, improved_content, improved_topic))
                    processed_segments.add(segment_key)
                    continue
            
            logging.warning(f"Failed to process segment {start_str}-{end_str} with ")
        
        except Exception as e:
            logging.error(f"Error processing segment {start_str}-{end_str}: {str(e)}")
        
        improved_topic = clean_topic(topic)
        improved_segments.append((start_str, end_str, text, improved_topic))
        processed_segments.add(segment_key)
    
    # Format as proper Markdown with single timestamp
    result = ""
    for start_str, end_str, text, topic in improved_segments:
        result += f"**[{start_str} - {end_str}] {topic}**\n\n{text}\n\n---\n\n"
    
    return result

class TranscriptApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Timecode Analyzer")
        self.root.geometry("800x600")
        self.queue = Queue()
        self.font_size = tk.StringVar(value="10")
        self.current_html = ""  # Store current HTML content
        self.current_transcript = ""  # Store current transcript text for copying
        
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=9)  # Output frame takes 90% height
        
        # Input frame
        self.input_frame = ttkb.Frame(self.root)
        self.input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)
        
        self.url_label = ttkb.Label(self.input_frame, text="YouTube URL:")
        self.url_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.url_entry = ttkb.Entry(self.input_frame)
        self.url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.analyze_button = ttkb.Button(self.input_frame, text="Analyze Transcript", style="primary.TButton", command=self.start_analysis)
        self.analyze_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.copy_button = ttkb.Button(self.input_frame, text="Copy All", style="secondary.TButton", command=self.copy_transcript)
        self.copy_button.grid(row=0, column=3, padx=5, pady=5)
        
        self.font_label = ttkb.Label(self.input_frame, text="Font Size:")
        self.font_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        
        self.font_combobox = ttkb.Combobox(self.input_frame, textvariable=self.font_size, values=["8", "10", "12", "14", "16"], width=5)
        self.font_combobox.grid(row=0, column=5, padx=5, pady=5)
        self.font_combobox.bind("<<ComboboxSelected>>", self.change_font_size)
        
        # Progress bar
        self.progress = ttkb.Progressbar(self.root, mode="indeterminate", style="info.TProgressbar")
        self.progress.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.progress.grid_remove()
        
        # Output frame with HtmlFrame for Markdown
        self.output_frame = ttkb.Frame(self.root)
        self.output_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_rowconfigure(0, weight=1)
        
        self.output_html = HtmlFrame(self.output_frame, messages_enabled=False)
        self.output_html.grid(row=0, column=0, sticky="nsew")
        
        # Log frame
        self.log_frame = ttkb.Frame(self.root)
        self.log_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.log_frame.grid_columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(self.log_frame, wrap="word", height=5, font=("Helvetica", 10))
        self.log_text.grid(row=0, column=0, sticky="ew")
        self.log_text.config(state="disabled")
        
        self.log_scrollbar = ttkb.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)
        
        # Status label
        self.status_label = ttkb.Label(self.root, text="Ready", style="info.TLabel")
        self.status_label.grid(row=4, column=0, padx=10, pady=5, sticky="sw")
        
        # Check queues for messages
        self.root.after(100, self.check_queues)
    
    def copy_transcript(self):
        if self.current_transcript:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_transcript)
            messagebox.showinfo("Success", "Transcript copied to clipboard!")
        else:
            messagebox.showwarning("Warning", "No transcript available to copy.")
    
    def change_font_size(self, event=None):
        size = int(self.font_size.get())
        # Update HTML with new font size
        if self.current_html:
            self.output_html.load_html(f'<style>body {{ font-size: {size}px; }}</style>' + self.current_html)
        self.log_text.config(font=("Helvetica", size))
    
    def start_analysis(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
        
        # Disable UI elements during processing
        self.analyze_button.config(state="disabled")
        self.url_entry.config(state="disabled")
        self.copy_button.config(state="disabled")
        self.font_combobox.config(state="disabled")
        self.output_html.load_html("")
        self.current_html = ""
        self.current_transcript = ""
        self.status_label.config(text="Processing...")
        self.progress.grid()
        self.progress.start()
        
        # Run analysis in a separate thread
        threading.Thread(target=self.analyze_transcript, args=(url,), daemon=True).start()
    
    def analyze_transcript(self, url):
        try:
            transcript_groups = get_transcript(url)
            improved_transcript = post_process_with_gemma(transcript_groups)
            self.queue.put(("success", improved_transcript))
        except Exception as e:
            self.queue.put(("error", str(e)))
    
    def check_queues(self):
        # Check transcript queue
        try:
            msg_type, msg = self.queue.get_nowait()
            if msg_type == "success":
                # Store transcript text for copying
                self.current_transcript = msg
                # Convert Markdown to HTML and display
                self.current_html = markdown(msg)
                self.output_html.load_html(f'<style>body {{ font-size: {self.font_size.get()}px; }}</style>' + self.current_html)
                self.status_label.config(text="Analysis Complete")
            elif msg_type == "error":
                messagebox.showerror("Error", f"Failed to process transcript: {msg}")
                self.status_label.config(text="Error")
        except Empty:
            pass
        
        # Check log queue
        try:
            log_msg = log_queue.get_nowait()
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, log_msg + "\n")
            self.log_text.config(state="disabled")
            self.log_text.see(tk.END)
        except Empty:
            pass
        
        # Re-enable UI elements
        self.analyze_button.config(state="normal")
        self.url_entry.config(state="normal")
        self.copy_button.config(state="normal")
        self.font_combobox.config(state="normal")
        self.progress.stop()
        self.progress.grid_remove()
        
        # Continue checking queues
        self.root.after(100, self.check_queues)

def main():
    root = ttkb.Window(themename="darkly")
    app = TranscriptApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()