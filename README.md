# RAG-Based AI Research Assistant
> Efficiently Discover, Analyze, and Summarize Research Papers Using Retrieval-Augmented Generation (RAG) and Modern NLP Techniques.

---

## Introduction

The exponential growth of academic papers makes it increasingly challenging for researchers to stay updated. We aims to alleviate this burden by automating:

- **Fetching relevant papers from arXiv**
- **Summarizing and comparing papers via a RAG-powered chat assistant**
- **Visualizing relationships between papers through TF-IDF similarity matrices**

By leveraging cutting-edge AI and information retrieval techniques, DEEPLOST helps streamline literature reviews and research exploration.

---

## Features

- **Topic-Based Paper Fetching**: Retrieve papers from arXiv with simple keyword queries.
- **Interactive Chat**: Summarize, compare, and query papers using a Retrieval-Augmented Generation (RAG) system.
- **Similarity Visualization**: Generate and view a TF-IDF-based heatmap of paper similarities.
- **Customizable & Open**: Modify embedding models, prompts, or expand sources easily.
- **User-Friendly Interface**: Powered by Gradio; no complex setup required.

---

## System Architecture

```plaintext
User Input (Topic, Paper Count)
        ↓
Query arXiv → Download PDFs
        ↓
➤ TF-IDF Pipeline (scikit-learn)
    • Text Extraction
    • Cleaning
    • TF-IDF Vectorization
    • Cosine Similarity Computation
    • Heatmap Visualization
➤ RAG Pipeline (LangChain + FAISS + OpenAI)
    • Text Extraction
    • Chunking
    • Embedding Generation
    • Vector Storage
    • Retrieval and Chat
        ↓
Results Display (Gradio Interface)
```
## Prerequisites
- Python 3.8+
- Azure OpenAI API key
---
## Maintainers
- Danlei Zhu (dz2t@mtmail.mtsu.edu)
- Yousaf Khaliq (yak2a@mtmail.mtsu.edu)
- Haoyuan Wang (hw4m@mtmail.mtsu.edu)
---
## A.I. Disclaimer: Work for this project was completed with the aid of artificial intelligence tools.
