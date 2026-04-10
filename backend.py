import os
import time
import json
import warnings
import numpy as np
from pathlib import Path
from io import BytesIO
from typing import List, Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-..................")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
HF_TOKEN        = os.getenv("HF_TOKEN", "")   # ← ton token HuggingFace ici

# Chemin vers le modèle .h5 entraîné sur HAM10000
CNN_MODEL_PATH  = os.getenv("CNN_MODEL_PATH", "model_sante_classique.h5")
# Classes produites par le notebook (ordre alphabétique Keras)
CNN_CLASSES     = ["benin", "malin"]

FAISS_INDEX_PATH = "rag_index.faiss"
RAG_DOCS_PATH    = "rag_docs.json"

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="MedAI Clinical Backend",
    description="Skin lesion CNN | Multimodal HuggingFace | Medical RAG + GPT",
    version="2.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class TextInput(BaseModel):
    text: str

class MultimodalMessage(BaseModel):
    role: str
    content: str

class MultimodalInput(BaseModel):
    messages: List[MultimodalMessage]
    new_message: str

# =============================================================================
# LAZY MODEL LOADERS
# =============================================================================

_cnn_model          = None
_hf_pipeline        = None
_embedding_model    = None
_faiss_index        = None
_rag_docs           = None
_openai_client      = None


def load_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        if not Path(CNN_MODEL_PATH).exists():
            return None
        try:
            from tensorflow.keras.models import load_model
            _cnn_model = load_model(CNN_MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Cannot load CNN model: {e}")
    return _cnn_model


def load_hf_pipeline():
    """
    BLIP VQA Large — meilleur que base, répond aux questions sur les images médicales.
    Nécessite un token HuggingFace (défini dans HF_TOKEN).
    ~1.5 GB RAM, fonctionne sur CPU.
    """
    global _hf_pipeline
    if _hf_pipeline is None:
        import torch
        from transformers import BlipProcessor, BlipForQuestionAnswering
        from huggingface_hub import login

        if HF_TOKEN:
            login(token=HF_TOKEN)
            print("[HF] Authentifié avec le token HF.")

        print("[HF] Loading Salesforce/blip-vqa-large ...")
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-large",
            token=HF_TOKEN or None
        )
        model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-large",
            torch_dtype=torch.float32,
            token=HF_TOKEN or None
        )
        model.eval()
        _hf_pipeline = {"processor": processor, "model": model}
        print("[HF] Model ready.")
    return _hf_pipeline


def load_rag_components():
    global _embedding_model, _faiss_index, _rag_docs
    from sentence_transformers import SentenceTransformer
    import faiss

    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if _faiss_index is None:
        if Path(FAISS_INDEX_PATH).exists():
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(RAG_DOCS_PATH, "r", encoding="utf-8") as f:
                _rag_docs = json.load(f)
        else:
            _faiss_index, _rag_docs = _build_rag_index(_embedding_model)

    return _embedding_model, _faiss_index, _rag_docs


def load_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# =============================================================================
# RAG — KNOWLEDGE BASE BUILDER
# =============================================================================

def _build_rag_index(embedding_model=None):
    """
    Construit l'index FAISS depuis les fichiers .txt du dossier rag_docs/.
    Chaque fichier est découpé en chunks de ~800 caractères avec chevauchement 200.

    """
    import faiss

    RAG_DOCS_DIR = Path("rag_docs")
    CHUNK_SIZE   = 800    # caractères par chunk
    CHUNK_OVERLAP = 200   # chevauchement entre chunks

    def _chunk_text(text: str, source: str) -> list:
        """Découpe un texte en chunks avec chevauchement."""
        chunks = []
        start  = 0
        text   = text.strip()
        while start < len(text):
            end   = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 100:  # ignorer les mini-fragments
                chunks.append({"source": source, "content": chunk})
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ── Charger tous les .txt du dossier rag_docs/ ──────────────────────
    knowledge_base = []
    doc_id = 0

    if RAG_DOCS_DIR.exists():
        txt_files = sorted(RAG_DOCS_DIR.glob("*.txt"))
        for txt_path in txt_files:
            try:
                text = txt_path.read_text(encoding="utf-8")
                # Titre = première ligne non vide
                first_line = next(
                    (l.strip() for l in text.splitlines() if l.strip()), txt_path.stem
                )
                chunks = _chunk_text(text, first_line)
                for chunk in chunks:
                    knowledge_base.append({
                        "id":      doc_id,
                        "title":   first_line,
                        "source":  txt_path.name,
                        "content": chunk["content"]
                    })
                    doc_id += 1
            except Exception as e:
                print(f"[RAG] Warning: could not load {txt_path.name}: {e}")
    
    # ── Fallback si aucun fichier trouvé ────────────────────────────────
    if not knowledge_base:
        print("[RAG] No .txt files found in rag_docs/. Using minimal fallback.")
        knowledge_base = [
            {
                "id": 0,
                "title": "Notice",
                "source": "fallback",
                "content": (
                    "Aucun document médical chargé. "
                    "Créer le dossier rag_docs/ et y placer des fichiers .txt "
                    "pour enrichir la base de connaissances RAG."
                )
            }
        ]

    # ── Encoder et indexer ───────────────────────────────────────────────
    texts      = [doc["content"] for doc in knowledge_base]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(RAG_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    print(f"[RAG] Index built: {len(knowledge_base)} chunks from {doc_id} segments")
    return index, knowledge_base


# =============================================================================
# MODULE 1 — CNN CLASSIFICATION (HAM10000 .h5)
# POST /cnn/predict
# =============================================================================

@app.post("/cnn/predict", summary="Skin lesion binary classification (benin / malin)")
async def cnn_predict(file: UploadFile = File(...)):
    """
    Charge le modèle Keras .h5 entraîné sur HAM10000.
    Input  : image JPG/PNG
    Output : classe (bénin/malin) + probabilité
    """
    t0 = time.time()

    model = load_cnn_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"CNN model not found at '{CNN_MODEL_PATH}'. "
                   "Copy model_sante_classique.h5 next to backend.py."
        )

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB").resize((128, 128))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)   # (1, 128, 128, 3)

        prob = float(model.predict(arr, verbose=0)[0][0])

        # Le notebook encode : benin=0, malin=1 (ordre alphabétique Keras)
        predicted_class = CNN_CLASSES[int(prob > 0.5)]
        confidence = prob if predicted_class == "malin" else (1.0 - prob)

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "raw_score": round(prob, 4),
            "execution_ms": round((time.time() - t0) * 1000, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


# =============================================================================
# MODULE 2 — HUGGINGFACE MULTIMODAL (VQA — image + texte)
# POST /hf/analyze          → analyse une image avec une question
# POST /hf/chat             → conversation texte général lié au patient
# =============================================================================

@app.post("/hf/analyze", summary="Medical image analysis — GPT-4o Vision")
async def hf_analyze(
    question: str = "Is this skin lesion benign or malignant? Describe its characteristics.",
    file: UploadFile = File(...)
):
    """
    GPT-4o Vision — analyse clinique des images dermoscopiques.
    """
    import base64, torch
    t0 = time.time()
    try:
        client   = load_openai()
        contents = await file.read()

        b64_img = base64.b64encode(contents).decode("utf-8")
        ext     = (file.filename or "image.jpg").rsplit(".", 1)[-1].lower()
        mime    = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are an expert dermatologist assistant. "
                    "Analyze dermoscopy and clinical skin images with clinical precision. "
                    "Describe: color variation, border irregularity, asymmetry, "
                    "dermoscopic structures, and give a differential diagnosis. "
                    "Be concise and clinically rigorous."
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{b64_img}",
                        "detail": "high"
                    }}
                ]}
            ],
            max_tokens=500,
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()

        return {
            "question":     question,
            "answer":       answer,
            "confidence":   1.0,
            "model":        "gpt-4o-vision",
            "execution_ms": round((time.time() - t0) * 1000, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-4o vision error: {e}")


@app.post("/hf/chat", summary="Text-only medical chat (HuggingFace zero-shot)")
async def hf_chat(input: TextInput):
    """
    Classification zero-shot sur une requête patient en langage naturel.
    Retourne les pathologies les plus probables selon le texte saisi.
    """
    t0 = time.time()
    try:
        from transformers import pipeline as hf_pipe
        classifier = hf_pipe(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        candidate_labels = [
            "Flu", "Common Cold", "Pneumonia", "COVID-19",
            "Diabetes", "Hypertension", "Allergy",
            "Skin infection", "Dermatitis"
        ]
        result = classifier(input.text, candidate_labels)

        top3 = [
            {"label": lbl, "score": round(sc, 4)}
            for lbl, sc in zip(result["labels"][:3], result["scores"][:3])
        ]

        return {
            "query": input.text,
            "top_predictions": top3,
            "model": "facebook/bart-large-mnli",
            "execution_ms": round((time.time() - t0) * 1000, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace chat error: {e}")


# =============================================================================
# MODULE 3 — RAG + GPT-3.5 (médecine générale)
# POST /rag/query
# =============================================================================

@app.post("/rag/query", summary="Medical RAG — retrieve + generate (GPT-3.5)")
def rag_query(input: TextInput):
    """
    1. Encode la requête patient avec sentence-transformers (all-MiniLM-L6-v2)
    2. Recherche FAISS les 3 documents médicaux les plus proches
    3. Injecte les documents dans le prompt GPT-3.5
    4. Retourne la réponse ancrée dans la base de connaissances
    """
    t0 = time.time()
    try:
        embedding_model, faiss_index, docs = load_rag_components()

        # 1. Embed
        q_emb = np.array(embedding_model.encode([input.text])).astype("float32")

        # 2. Search
        distances, indices = faiss_index.search(q_emb, k=3)
        retrieved = [docs[i] for i in indices[0] if i < len(docs)]

        # 3. Build context
        context = "\n\n".join(
            f"[{doc['title']}]\n{doc['content']}"
            for doc in retrieved
        )

        # 4. Generate
        try:
            client = load_openai()
            prompt = (
                "Tu es un assistant médical clinique. "
                "Réponds exclusivement à partir du contexte fourni. "
                "Si l'information n'est pas dans le contexte, dis-le clairement. "
                "Rappelle systématiquement qu'une consultation médicale reste nécessaire.\n\n"
                f"CONTEXTE CLINIQUE :\n{context}\n\n"
                f"QUESTION PATIENT : {input.text}\n\n"
                "Réponse (3-4 phrases, ton clinique, sans jargon excessif) :"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            answer = response.choices[0].message.content.strip()

        except Exception:
            # Fallback sans LLM
            answer = retrieved[0]["content"][:500] + "..." if retrieved else "No context found."

        return {
            "query": input.text,
            "answer": answer,
            "sources": [
                {
                    "title": doc["title"],
                    "relevance": round(float(1 / (1 + distances[0][i])), 3)
                }
                for i, doc in enumerate(retrieved)
            ],
            "execution_ms": round((time.time() - t0) * 1000, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", summary="Backend health check")
def health():
    cnn_ready = Path(CNN_MODEL_PATH).exists()
    rag_ready = Path(FAISS_INDEX_PATH).exists()
    return {
        "status": "ok",
        "cnn_model_ready": cnn_ready,
        "rag_index_ready": rag_ready,
        "cnn_model_path": CNN_MODEL_PATH,
        "llm_model": LLM_MODEL
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)