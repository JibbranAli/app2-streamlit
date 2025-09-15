# video_summarizer_ultimate.py

import streamlit as st
import os
import tempfile
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import numpy as np
import librosa
import torch
import pyttsx3
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from googletrans import Translator

# -------------------- App Config --------------------
st.set_page_config(page_title="AI Video Summarizer Ultimate", layout="wide")
st.title("🎬 AI Video Summarizer Ultimate")
st.write("Upload a video and get multi-level summaries, highlight reels, and more!")

# -------------------- Video Upload --------------------
video_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    st.video(video_path)

    # -------------------- Whisper Model --------------------
    st.info("⏳ Transcribing video using Whisper...")
    model = whisper.load_model("base")  # tiny, base, small, medium, large
    result = model.transcribe(video_path)
    transcript = result["text"]

    st.subheader("📝 Full Transcription")
    st.text_area("Transcript", transcript, height=300)

    # -------------------- Multi-Level Summaries --------------------
    st.info("🧠 Generating summaries...")
    summarizer = pipeline("summarization")
    
    # Short Summary
    short_summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    st.subheader("🟢 Short Summary")
    st.write(short_summary)

    # Ultra-Short Summary
    ultra_short_summary = summarizer(transcript, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    st.subheader("🔵 Ultra-Short Summary")
    st.write(ultra_short_summary)

    # -------------------- Keyword & Sentiment Analysis --------------------
    st.info("🔍 Analyzing text & sentiment...")
    nlp = pipeline("sentiment-analysis")
    sentiment = nlp(transcript)[0]
    st.write(f"**Sentiment:** {sentiment['label']} with score {sentiment['score']:.2f}")

    # Keywords
    words = [word.lower() for word in transcript.split()]
    common_words = Counter(words).most_common(10)
    st.write("**Top Keywords:**")
    st.write([w[0] for w in common_words])

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # -------------------- Emotion Detection (Optional: simple) --------------------
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    emotions = emotion_classifier(transcript)[0]
    emotion_scores = {e['label']: e['score'] for e in emotions}
    st.write("**Emotion Scores:**", emotion_scores)

    # -------------------- AI Voiceover --------------------
    st.info("🎤 Generating AI Voiceover for summary...")
    tts_engine = pyttsx3.init()
    tts_engine.save_to_file(short_summary, "summary_voice.mp3")
    tts_engine.runAndWait()
    st.audio("summary_voice.mp3")

    # -------------------- Audio Energy & Highlight Video --------------------
    st.info("🎥 Creating AI Highlight Video...")
    y, sr = librosa.load(video_path, sr=None)
    hop_length = 512
    energy = np.array([sum(abs(y[i:i+hop_length]**2)) for i in range(0, len(y), hop_length)])
    top_peaks = energy.argsort()[-3:]  # top 3 high-energy segments

    clip = VideoFileClip(video_path)
    highlight_clips = []
    for peak in top_peaks:
        start = max(0, peak * hop_length / sr - 2)  # 2 sec before peak
        end = min(clip.duration, start + 4)         # 4 sec clip
        subclip = clip.subclip(start, end)
        highlight_clips.append(subclip)

    final_highlight = concatenate_videoclips(highlight_clips)
    highlight_path = "highlight.mp4"
    final_highlight.write_videofile(highlight_path, codec="libx264")
    st.video(highlight_path)
    st.success("✅ Short video summary generated successfully!")

    # -------------------- Multi-Language Support --------------------
    st.info("🌐 Translating summaries...")
    translator = Translator()
    lang_choice = st.selectbox("Translate Summary to:", ["English", "Hindi", "Spanish", "French"])
    if lang_choice != "English":
        translated_summary = translator.translate(short_summary, dest=lang_choice[:2].lower()).text
        st.subheader(f"🗣 Translated Summary ({lang_choice})")
        st.write(translated_summary)

    # -------------------- Dashboard / Interactive Timeline --------------------
    st.info("📊 Interactive Timeline (Audio Peaks)")
    timeline_fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(energy, color='purple')
    ax.set_title("Audio Energy Timeline")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Energy")
    for peak in top_peaks:
        ax.axvline(x=peak, color='red', linestyle='--')
    st.pyplot(timeline_fig)
