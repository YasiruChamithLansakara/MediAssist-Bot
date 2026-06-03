# MediAssist Bot — Evaluation Report

> Generated: 2026-06-03 18:26:55

## Overall Score: **90.5%**

| Component | Score | Grade | Status |
|---|---|---|---|
| Dataset | 59.3% | POOR | WARN |
| Drug Lookup | 100.0% | EXCELLENT | PASS |
| Ner | 76.2% | GOOD | PASS |
| Safety | 100.0% | EXCELLENT | PASS |
| Conversation Memory | 100.0% | EXCELLENT | PASS |
| Chat Engine | 100.0% | EXCELLENT | PASS |
| Rag | 0.0% | — | SKIP |
| Activity Metrics | 100.0% | EXCELLENT | PASS |
| Ocr | 100.0% | EXCELLENT | WARN |
| Faiss | 60.0% | FAIR | PASS |
| Llm | 100.0% | EXCELLENT | PASS |
| Pytest | 100.0% | EXCELLENT | PASS |

## Component Details

### Dataset Quality
- Total drugs: **3869**
- Avg field coverage: **59.3%**
- Duplicates: **0**

### Drug Lookup
- Accuracy: **100.0%** (20/20 passed)
- Avg latency: **11.4 ms**

### NER Drug Extraction
- Precision: **66.7%**
- Recall: **88.9%**
- F1 Score: **76.2%**

### Safety Layer
- Emergency recall: **100.0%**
- Safe-phrase precision: **100.0%**

### OCR Pipeline
- Engines available: **none**
- Text analysis hit rate: **100.0%**
- Preprocessing score: **100.0%**
- Avg latency: **222.4 ms**

### FAISS Semantic Search
- Vectors indexed: **3869**
- Semantic hit rate: **60.0%**
- Avg query latency: **26.0 ms**

### Automated Tests (pytest)
- Passed: **97/97**
- Pass rate: **100.0%**

---
*MediAssist Bot — AI-powered medication assistant for chronic disease patients*