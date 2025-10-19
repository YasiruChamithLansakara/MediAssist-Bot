```markdown
# 🧠 MediAssist Bot
### An AI-powered assistant that reads doctor prescriptions, extracts medicine names, and provides intelligent medical guidance.

---

## 🚀 Project Overview
**MediAssist Bot** is an AI-based system designed to help users understand medical prescriptions.  
It can:
1. Read **handwritten or typed prescriptions** using OCR.
2. Extract **medicine names** using NLP techniques.
3. Provide **AI-powered advice** (like medicine information, side effects, and interactions).
4. Allow users to **chat** with the bot for further questions.

---

## 🧩 System Architecture
```

[Prescription Image]
↓
OCR Module (image → text)
↓
Medicine Extraction (NER / Matching)
↓
AI Advice Engine (transformers pipeline)
↓
Chat Interface (Streamlit / Flask)

```

---

## 🗂️ Project Structure
```

MediAssistBot/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── medicine_list.csv
│   └── sample_prescriptions/
│
├── notebooks/
│   ├── 01_OCR_testing.ipynb
│   ├── 02_Medicine_Extraction.ipynb
│   └── 03_AI_Advice_Testing.ipynb
│
├── mediassist/
│   ├── **init**.py
│   ├── ocr_module.py
│   ├── medicine_extractor.py
│   ├── ai_advice.py
│   ├── chat_interface.py
│   ├── utils.py
│   └── config.py
│
├── webapp/
│   ├── app.py
│   ├── templates/
│   ├── static/
│
├── tests/
│   ├── test_ocr.py
│   ├── test_medicine_extractor.py
│   └── test_ai_advice.py
│
└── scripts/
├── run_ocr.py
├── run_extraction.py
└── run_chat.py

````

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YasiruChamithLansakara/MediAssist-Bot.git
cd MediAssistBot
````

### 2️⃣ Create Virtual Environment (Optional)

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

### 🔹 Step 1: OCR Module

* Converts uploaded prescription images into readable text.
* Uses **OpenCV + PyTesseract + EasyOCR**.

### 🔹 Step 2: Medicine Extraction

* Identifies medicine names using **spaCy** or **transformer-based models**.
* Can also use a predefined **medicine list** for matching.

### 🔹 Step 3: AI Advice

* Uses **Hugging Face transformers pipeline** for contextual responses.
* Provides **medicine usage, side effects, and warnings**.

### 🔹 Step 4: Chat Interface

* Simple interface built with **Streamlit or Flask**.
* Lets users upload images and chat with the bot.

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

## 👥 Team Roles (Suggested)

| Member   | Responsibility             |
| -------- | -------------------------- |
| Member 1 | OCR Module & Preprocessing |
| Member 2 | Medicine Extraction (NLP)  |
| Member 3 | AI Advice & Chat Interface |
| Member 4 | Integration & Testing      |

---

## 📦 Dependencies

See [`requirements.txt`](./requirements.txt) for the full list.
Key packages:

* **OpenCV, PyTesseract, EasyOCR** → image to text
* **spaCy, transformers, torch** → NLP & AI
* **Flask / Streamlit** → web chat interface
* **requests, dotenv** → API connections and configuration

---

## ⚠️ Disclaimer

MediAssist Bot is for **educational and informational purposes only**.
It **does not replace professional medical advice or diagnosis**.

---

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ⭐ Contribution Guidelines

1. Create a new branch for your feature.
2. Commit descriptive messages.
3. Submit a pull request for review.
4. Keep code modular and documented.

---

### 🩺 Developed with ❤️ by Team MediAssist

```
