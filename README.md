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
MediAssistBot/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── raw/
│ ├── processed/
│ ├── medicine_list.csv
│ └── sample_prescriptions/
│
├── notebooks/
│ ├── 01_OCR_testing.ipynb
│ ├── 02_Medicine_Extraction.ipynb
│ └── 03_AI_Advice_Testing.ipynb
│
├── mediassist/
│ ├── init.py
│ ├── ocr_module.py
│ ├── medicine_extractor.py
│ ├── ai_advice.py
│ ├── chat_interface.py
│ ├── utils.py
│ └── config.py
│
├── webapp/
│ ├── app.py
│ ├── templates/
│ └── static/
│
├── tests/
│ ├── test_ocr.py
│ ├── test_medicine_extractor.py
│ └── test_ai_advice.py
│
└── scripts/
├── run_ocr.py
├── run_extraction.py
└── run_chat.py
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YasiruChamithLansakara/MediAssist-Bot.git
cd MediAssistBot
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
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

- **Programming:** Python, NumPy, Pandas, matplotlib, seaborn, scikit-learn 
- **OCR:** DeepseekOCR  
- **NLP / Transformers:** BioBERT  
- **Vector Database:** FAISS  
- **Knowledge Retrieval:** RAG (Retrieval-Augmented Generation)  
- **Backend:** FastAPI  
- **UI:** Streamlit  

---

## 🧪 Example Usage
```bash
# Run OCR on a prescription
python scripts/run_ocr.py

# Extract medicines from text
python scripts/run_extraction.py

# Launch the chat interface
streamlit run webapp/app.py
```

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
