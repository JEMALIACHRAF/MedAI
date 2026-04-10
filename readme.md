
# MedAI Clinical — AI-Powered Dermatology Platform

> Research platform combining CNN-based lesion classification, GPT-4o Vision multimodal analysis, and retrieval-augmented generation over a curated medical knowledge base.

---

## Overview

MedAI Clinical is a full-stack AI research platform built around three independent clinical modules, all exposed through a modern HTML/JS frontend and a FastAPI backend.

| Module | Endpoint | Model |
|---|---|---|
| Lesion Classification | `POST /cnn/predict` | Custom CNN · HAM10000 |
| Multimodal VQA | `POST /hf/analyze` | GPT-4o Vision |
| Text Classification | `POST /hf/chat` | facebook/bart-large-mnli |
| Clinical RAG | `POST /rag/query` | all-MiniLM-L6-v2 + GPT-3.5 |

---

## Project Structure

```
medai-clinical/
├── backend.py               # FastAPI application
├── index.html               # Landing page
├── frontend.html            # Platform interface
├── model_sante_classique.h5 # Trained CNN model (HAM10000)
├── rag_docs/                # Medical knowledge base (.txt files)
│   └── *.txt
├── rag_index.faiss          # Auto-generated FAISS index (first run)
└── rag_docs.json            # Auto-generated doc store (first run)
```

---

## Requirements

- Python 3.10+
- A modern browser (Chrome recommended)
- OpenAI API key
- HuggingFace token (optional — for private models)

---

## Installation

```bash
pip install fastapi uvicorn openai transformers torch \
            faiss-cpu sentence-transformers pillow tensorflow numpy
```

---

## Configuration

Set environment variables before launching:

```bash
# Linux / macOS
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...          # optional
export CNN_MODEL_PATH=model_sante_classique.h5  # default value

# Windows
set OPENAI_API_KEY=sk-...
set HF_TOKEN=hf_...
```

The backend will fall back to the hardcoded values in `backend.py` if variables are not set.

---

## Launch

**Step 1 — Start the backend**

```bash
uvicorn backend:app --reload --port 8000
```

Check it's running at `http://localhost:8000/docs`.

**Step 2 — Open the frontend**

Double-click `index.html` in your file explorer. It opens directly in the browser — no build step, no Node.js required.

From the landing page, click **"Launch Platform"** to access the full interface.

---

## Modules

### 1. Lesion Classification (CNN)

Classifies dermoscopy images as **benign** or **malignant** using a convolutional neural network trained on the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) dataset.

- Input: JPG/PNG image (128×128 resized internally)
- Output: prediction class, confidence %, raw score, latency ms
- Model file: `model_sante_classique.h5` must be present at the project root

### 2. Multimodal Analysis

**VQA (Image + Question)** — GPT-4o Vision analyzes dermoscopy images against a free-form clinical question. Describes color variation, border irregularity, asymmetry, and provides a differential diagnosis.

**Text Classification** — Zero-shot classification of a patient symptom description using `facebook/bart-large-mnli`. Returns top-3 most likely pathologies from a predefined 9-label taxonomy.

### 3. Clinical RAG

Retrieval-augmented generation over `.txt` medical documents placed in the `rag_docs/` folder.

Pipeline:
1. Embed query with `all-MiniLM-L6-v2`
2. Retrieve top-3 chunks via FAISS (L2 distance)
3. Inject context into GPT-3.5 prompt
4. Return grounded answer + source relevance scores

The FAISS index (`rag_index.faiss`) and document store (`rag_docs.json`) are built automatically on first run from the `.txt` files in `rag_docs/`.

---

## API Reference

### `GET /health`
Returns backend status, CNN model availability, and RAG index status.

```json
{
  "status": "ok",
  "cnn_model_ready": true,
  "rag_index_ready": true,
  "llm_model": "gpt-4o-mini"
}
```

### `POST /cnn/predict`
Multipart form — field `file` (image).

```json
{
  "filename": "lesion.jpg",
  "prediction": "malin",
  "confidence": 0.914,
  "raw_score": 0.914,
  "execution_ms": 87.3
}
```

### `POST /hf/analyze`
Multipart form — field `file` (image) + query param `question`.

```json
{
  "question": "Is this benign or malignant?",
  "answer": "The lesion shows asymmetry and irregular borders...",
  "confidence": 1.0,
  "model": "gpt-4o-vision",
  "execution_ms": 1243.7
}
```

### `POST /hf/chat`
JSON body `{ "text": "..." }`.

```json
{
  "top_predictions": [
    { "label": "Flu", "score": 0.623 },
    { "label": "Common Cold", "score": 0.198 },
    { "label": "Allergy", "score": 0.089 }
  ],
  "model": "facebook/bart-large-mnli",
  "execution_ms": 542.1
}
```

### `POST /rag/query`
JSON body `{ "text": "..." }`.

```json
{
  "answer": "Type 2 diabetes is diagnosed when...",
  "sources": [
    { "title": "Diabetes Guidelines", "relevance": 0.891 },
    { "title": "Endocrinology Handbook", "relevance": 0.743 }
  ],
  "execution_ms": 389.4
}
```

---

## Common Issues

**CORS error in browser**

When opening HTML files from the filesystem (`file://`), some browsers send `Origin: null`. Fix by adding `"null"` to the backend origins:

```python
allow_origins=["*", "null"]
```

**CNN model not found**

Place `model_sante_classique.h5` at the project root, or override the path:

```bash
export CNN_MODEL_PATH=/path/to/your/model.h5
```

**RAG returns fallback answer**

Create the `rag_docs/` folder and add at least one `.txt` file. The index rebuilds automatically on the next backend start.

**HuggingFace models slow on first run**

`Salesforce/blip-vqa-large` (~1.5 GB) and `facebook/bart-large-mnli` (~1.6 GB) are downloaded on first use and cached locally. Subsequent runs are fast.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI · Uvicorn · Pydantic |
| CNN inference | TensorFlow / Keras · NumPy · Pillow |
| Vision LLM | OpenAI GPT-4o (high detail) |
| Text generation | OpenAI GPT-3.5-turbo |
| Multimodal model | Salesforce/blip-vqa-large |
| Zero-shot NLP | facebook/bart-large-mnli |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector search | FAISS (IndexFlatL2) |
| Frontend | HTML · CSS · Vanilla JS |

---

## Disclaimer

> This platform is intended for **research and educational purposes only**. Model outputs do not constitute clinical diagnoses and must not be used as a substitute for professional medical evaluation. Always consult a licensed healthcare professional.

---

## License

MIT — see `LICENSE` for details.
