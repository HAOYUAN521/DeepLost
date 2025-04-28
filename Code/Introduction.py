# -*- coding: utf-8 -*-

!pip install pdfplumber pymupdf matplotlib wordcloud scikit-learn sentence-transformers PyPDF2 spacy gradio openai

!python -m spacy download en_core_web_md

import asyncio
import pdfplumber
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import spacy
import gradio as gr
from openai import AzureOpenAI

# ========== Azure OpenAI Configuration ==========
client = AzureOpenAI(
    api_key="Eaanf2HV3s7WEby7Xn7tZrW5IVrbafFhABGITqWYBy9CNdNwFVbfJQQJ99BDACHYHv6XJ3w3AAABACOG7P37",  # replace with your key
    api_version="2024-12-01-preview",
    azure_endpoint="https://sv-openai-research-group2.openai.azure.com/"  # replace with your endpoint
)
AZURE_DEPLOYMENT_NAME = "gpt-4-3"  # replace with your deployment name

# ========== Model Loading ==========
nlp_spacy = spacy.load("en_core_web_md")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ========== PDF Text Extraction ==========
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def extract_text_with_pymupdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_pdfreader(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

extract_methods = {
    "pdfplumber": extract_text_from_pdf,
    "pymupdf": extract_text_with_pymupdf,
    "pdfreader": extract_text_pdfreader,
}

# ========== Summary with Azure OpenAI ==========
def generate_summary_azure(text):
    if len(text) > 20000:
        text = text[:20000]
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes academic PDFs."},
                {"role": "user", "content": f"Summarize the following document:\n{text}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Azure OpenAI summary failed: {str(e)}"

# ========== Similarity Calculations ==========
def compute_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def compute_lsa_similarity(text1, text2, n_components=100):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    svd = TruncatedSVD(n_components=min(n_components, tfidf.shape[1]-1))
    lsa = svd.fit_transform(tfidf)
    return cosine_similarity([lsa[0]], [lsa[1]])[0][0]

def compute_word2vec_similarity(text1, text2):
    vec1 = nlp_spacy(text1).vector
    vec2 = nlp_spacy(text2).vector
    return cosine_similarity([vec1], [vec2])[0][0]

def compute_bert_similarity(text1, text2):
    embeddings = bert_model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# ========== WordCloud ==========
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig

# ========== Main Logic ==========
async def research_assistant_v2(pdf_file, method):
    extract_func = extract_methods[method]
    pdf_text = extract_func(pdf_file.name)

    summary_azure = generate_summary_azure(pdf_text)
    wordcloud_plot = generate_wordcloud(pdf_text)

    def sim_all(summary):
        return (
            f"{compute_tfidf_similarity(summary, pdf_text):.4f}",
            f"{compute_lsa_similarity(summary, pdf_text):.4f}",
            f"{compute_word2vec_similarity(summary, pdf_text):.4f}",
            f"{compute_bert_similarity(summary, pdf_text):.4f}",
        )

    az_sims = sim_all(summary_azure)

    return (
        summary_azure,
        pdf_text,
        wordcloud_plot,
        *az_sims
    )

# ========== Gradio UI ==========
def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 📚 AI Research Assistant: PDF Extraction & Text Analysis")

        with gr.Row():
            pdf_file = gr.File(label="📂 Upload PDF File")
            extract_method = gr.Dropdown(
                label="📌 Choose Extraction Method",
                choices=list(extract_methods.keys()),
                value="pdfplumber"
            )

        gr.Markdown("""
        **ℹ️ Extraction Method Info**
        - `pdfplumber`: Best for layout-preserved text extraction. May fail on image-based PDFs.
        - `pymupdf`: Fast and robust. Good for general-purpose text extraction.
        - `pdfreader`: Simple and reliable but may miss complex layout elements.
        """)

        analyze_btn = gr.Button("🚀 Start Analysis")

        output_summary_azure = gr.Textbox(label="☁️ Azure Summary", lines=6)
        output_text = gr.Textbox(label="📃 Extracted Full Text", lines=10)
        output_wordcloud = gr.Plot(label="☁️ Word Cloud")

        with gr.Row():
            gr.Markdown("### ☁️ Azure Similarity Scores")
        with gr.Row():
            output_tfidf_sim_az = gr.Textbox(label="TF-IDF")
            output_lsa_sim_az = gr.Textbox(label="LSA")
            output_word2vec_sim_az = gr.Textbox(label="Word2Vec")
            output_bert_sim_az = gr.Textbox(label="BERT")

        async def async_wrapper(f, m):
            return await research_assistant_v2(f, m)

        analyze_btn.click(
            fn=async_wrapper,
            inputs=[pdf_file, extract_method],
            outputs=[
                output_summary_azure,
                output_text,
                output_wordcloud,
                output_tfidf_sim_az,
                output_lsa_sim_az,
                output_word2vec_sim_az,
                output_bert_sim_az,
            ]
        )

    return demo

# Run Gradio App
if __name__ == "__main__":
    demo = gradio_ui()
    demo.launch(share=True)

