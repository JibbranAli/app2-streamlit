from transformers import pipeline
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googletrans import Translator

def generate_summaries(transcript):
    summarizer = pipeline("summarization")
    short_summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    ultra_short_summary = summarizer(transcript, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    return short_summary, ultra_short_summary

def extract_keywords(transcript, top_n=10):
    words = [word.lower() for word in transcript.split()]
    common_words = Counter(words).most_common(top_n)
    return [w[0] for w in common_words]

def generate_wordcloud(transcript):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt

def translate_text(text, dest_lang="hi"):
    translator = Translator()
    return translator.translate(text, dest=dest_lang).text
