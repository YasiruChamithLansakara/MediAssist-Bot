# 🧠 MediAssist Bot
### 💊 AI-Based Medication Assistant for Chronic Diseases

---

## 📌 Overview

**MediAssist Bot** is an academic **AI-based medication understanding system** designed to assist patients—especially those with **chronic diseases**—in understanding **doctor prescriptions**.

The system focuses on:
- Reading **handwritten or printed prescriptions**
- Extracting **medicine names and dosage information**
- Providing **safe, non-diagnostic explanations** using trusted medical data
- Supporting users through a **chat-based interface**

⚠️ This project is developed as a **Proof-of-Concept (PoC)** for educational purposes only.

---

## 🚨 Problem Statement

Chronic disease patients worldwide often struggle to understand handwritten doctor prescriptions. These prescriptions can be unclear not only to patients but sometimes even to pharmacists. Misunderstanding medicine names, dosages, and instructions may lead to incorrect medication usage and serious health risks.

Existing online platforms are often unreliable, not user-friendly, and lack proper medical grounding. Therefore, there is a need for a **technically improved AI-based system** that provides **clear, patient-friendly explanations** using **valid and trusted medical data sources**, without replacing healthcare professionals.

---

## 💡 Project Idea

An AI-powered assistant that helps patients understand:
- Medicines prescribed by doctors  
- Dosage patterns and instructions  
- General purpose, warnings, side effects, and contraindications  

The system uses:
- **OCR** for prescription text extraction  
- **NLP & Transformer models** for medication entity extraction  
- **Retrieval-Augmented Generation (RAG)** for grounded explanations  
- A **chat interface** for user interaction  

---

## 🎯 Target Chronic Diseases

- Diabetes  
- Hypertension  
- Asthma  
- Heart Disease  
- Arthritis  
- Migraine  

> These diseases require long-term medication usage and are common globally, with better public dataset availability.

---

## 👥 Target Audience

- Patients with chronic diseases  
- Elderly patients  
- Caregivers  
- AI students  
- Medical students  
- Pharmacy students  

---

## 🌍 Region

- **Sri Lanka (Academic Context)**

---

## 🧩 System Architecture

```text
User selects disease → Enter age
            ↓
Prescription Image / Text Query
            ↓
OCR → Text Extraction
            ↓
Text Cleaning & Parsing
            ↓
NLP → Medicine Name & Dosage Extraction
            ↓
Vector Database (FAISS) → Drug Lookup:
 - openFDA → dosage, warnings
 - DrugBank → drug class
 - Kaggle / MedDRA → high-level indication
            ↓ (RAG)
LLM → Safe explanation + disclaimer
            ↓
Chat Interface (UI)
```

---

## 🗂️ Project Structure

```text
MediAssist-Bot/
│
├── README.md
├── setup_linux.sh                  # one-command environment setup + verification
├── requirements.txt                # core API (pinned)
├── requirements-rag.txt            # retrieval: faiss, sentence-transformers, rank-bm25, easyocr
├── requirements-llm.txt            # groq / openai clients
├── requirements-nlp.txt            # optional biomedical NER (see Python-version note)
│
├── app/                            # FastAPI backend
│ ├── main.py                       # 11 endpoints, middleware, startup warmup
│ ├── ml/
│ │ ├── embeddings.py               # sentence-transformers → OpenAI → TF-IDF
│ │ └── faiss_store.py              # hybrid dense + BM25 retrieval over label sections
│ └── services/
│   ├── drug_lookup.py              # ingredient-identity matching over the drug CSV
│   ├── ner_service.py              # medication extraction (rules + optional SciSpaCy)
│   ├── ocr_service.py              # Tesseract + EasyOCR hybrid
│   ├── llm_service.py              # grounded answer generation
│   ├── safety_service.py           # emergency detection + disclaimers
│   ├── chat_engine.py              # orchestration, intent, off-topic guard
│   ├── conversation_memory.py      # per-session history
│   └── rag_service.py              # compatibility shim (SQLite RAG removed)
│
├── mediassist-frontend/            # React 19 + Vite single-page app
│ └── src/App.jsx                   # Lookup / Chat / Prescription views
│
├── data/
│ ├── raw/                          # openFDA, DrugBank, MedDRA sources (gitignored, ~11 GB)
│ ├── processed/
│ │ └── drug_knowledge_bot_ready_clean.csv   # canonical dataset — 3,881 drugs
│ └── faiss_index/                  # generated on startup (gitignored)
│
├── scripts/
│ ├── evaluate_pipeline.py          # 12-component scoring harness → HTML/JSON/MD
│ ├── add_missing_essential_drugs.py# backfills WHO-essential drugs from openFDA
│ └── …                             # dataset build and validation utilities
│
├── tests/                          # 99 tests: unit, integration, ocr
└── reports/                        # generated evaluation reports
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YasiruChamithLansakara/MediAssist-Bot.git
cd MediAssistBot
```

### 2️⃣ Set Up the Environment

**Linux / macOS** — one command builds `.venv/` and verifies every component:
```bash
./setup_linux.sh
./setup_linux.sh --check     # re-verify an existing environment
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-rag.txt -r requirements-llm.txt
```

> Install CPU-only PyTorch first. The default wheel pulls ~3 GB of CUDA
> libraries that this project never uses — EasyOCR runs with `gpu=False` and
> the embedding model runs on CPU.

Tesseract is a system package, not a pip package:
```bash
sudo apt install tesseract-ocr      # Debian/Ubuntu
brew install tesseract              # macOS
```
Without it the OCR pipeline still runs on EasyOCR alone.

### 3️⃣ Configure
```bash
cp .env.example .env       # then add LLM_API_KEY (free key: https://console.groq.com)
```

---

## 🧠 How It Works

### 1️⃣ OCR Processing
- Extracts text from prescription images
- Handles noisy and handwritten text

### 2️⃣ Medication Extraction (NLP)
- Identifies medicine names and dosage
- Uses transformer-based models

### 3️⃣ Knowledge Retrieval + LLM
- Retrieves verified drug information
- Generates grounded explanations with disclaimers

### 4️⃣ Chat Interface
- Allows follow-up questions
- Maintains conversation context

---

## 📊 Datasets Used

| Data Type | Source |
|---------|--------|
| Disease data | Kaggle |
| Medicine name | openFDA |
| Brand / Generic names | openFDA |
| Drug class | DrugBank |
| Purpose & indications | openFDA, DrugBank |
| Dosage patterns | openFDA, MedDRA |
| Side effects & warnings | openFDA, MedDRA |
| Contraindications | DrugBank, openFDA |
| Handwritten prescriptions | Kaggle / HF (synthetic & public) |

> All datasets are **public, trusted, and ethically approved**.

---

## 🛠️ Tools & Technologies

- **Programming:** Python 3.11+, NumPy, Pandas, scikit-learn
- **OCR:** Tesseract (binarised path) + EasyOCR (grayscale path), best candidate chosen by medication-signal score
- **Medication extraction:** rule-based n-grams validated against the drug knowledge base, with optional SciSpaCy `en_core_sci_sm`
- **Drug matching:** RapidFuzz with a salt-normalising ingredient-identity layer
- **Retrieval:** FAISS (`IndexFlatIP`, cosine) + BM25, fused with Reciprocal Rank Fusion, over section-level label chunks
- **Embeddings:** `all-MiniLM-L6-v2` locally, with OpenAI and TF-IDF fallbacks
- **LLM:** Groq `openai/gpt-oss-20b` (free tier), with a model-rotation chain
- **Backend:** FastAPI + Uvicorn
- **UI:** React 19 + Vite

> The original proposal named DeepseekOCR, BioBERT and Streamlit. Each was
> replaced during implementation — Tesseract/EasyOCR run without a GPU, the
> lookup-validated extractor measured better than transformer NER on this
> dataset, and the chat UI needed finer control than Streamlit allows.

---

## 🧪 Example Usage
```bash
# Start the API (http://127.0.0.1:8000, docs at /docs)
uvicorn app.main:app --reload

# Start the web UI (http://localhost:5173)
cd mediassist-frontend && npm install && npm run dev

# Run the test suite
pytest -q

# Score every pipeline component → reports/evaluation_report.html
python scripts/evaluate_pipeline.py

# Check the drug knowledge base for missing essential medicines
python scripts/add_missing_essential_drugs.py --dry-run
```

The vector index builds on first startup (~1 minute for 3,881 drugs) and is
cached in `data/faiss_index/`. It rebuilds automatically when the drug CSV
changes; delete the directory to force a rebuild.

---

## 📈 Current Evaluation

`python scripts/evaluate_pipeline.py`, 2026-09-09 — **overall 96.4%**

| Component | Score | Notes |
|---|---|---|
| Dataset quality | 99.9% | 3,881 drugs, 40/40 WHO-essential medicines for the six conditions |
| Drug lookup | 95.0% | ingredient-identity matching, 0.5 ms average |
| Medication extraction (NER) | 100.0% F1 | precision 100 / recall 100 over 15 prescription cases |
| Safety layer | 100.0% | emergency recall 100%, no false alarms on label questions |
| OCR pipeline | 100.0% | text-analysis hit rate |
| Hybrid retrieval | 86.7% | 30 queries: concept, exact-name and section-targeted |
| LLM answers | 100.0% grounding | answers only from retrieved evidence; refuses otherwise |
| Automated tests | 100.0% | 99/99 passing |

Scores are computed by `scripts/evaluate_pipeline.py` and written to
`reports/`. Retrieval and extraction sets are deliberately larger than the
number of cases needed to pass — a metric measured on five items cannot
distinguish a working component from a broken one.

---

## ⚠️ Safety, Ethics & Disclaimer

- For **educational and informational purposes only**
- No diagnosis, treatment, or medical decision-making
- Mandatory medical disclaimer included
- Explicit uncertainty handling
- No patient-identifiable data stored
- Secure handling of uploaded data

---

## ⏳ Project Timeline (46 Days)

| Phase | Activity | Duration |
|---|---|---|
| Phase 1 | Literature review & dataset preparation | 7 days |
| Phase 2 | OCR module development | 8 days |
| Phase 3 | NLP extraction & drug mapping | 10 days |
| Phase 4 | LLM & chat integration | 10 days |
| Phase 5 | Testing & evaluation | 7 days |
| Phase 6 | Documentation & presentation | 3 days |

---

## 📜 License

This project is released under the **MIT License**.

---

## ⭐ Contribution Guidelines

**1.** Create a new branch for your feature

**2.** Commit descriptive messages

**3.** Submit a pull request for review

**4.** Keep code modular and documented

---

## 🧬 Developed by  
**Team MediAssist ❤️**

© 2026 Team MediAssist. All Rights Reserved.
