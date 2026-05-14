# Explainable Graph-Based Hallucination Detection for Large Language Models

Rapid, accurate, and interpretable detection of hallucinated content in Large Language Model (LLM) outputs using graph learning, semantic analysis, and ensemble scoring.

## 1. Project Overview

This repository contains the implementation of our Major Project: **Explainable Graph-Based Hallucination Detection for Large Language Models (LLMs)**.

The project focuses on identifying factual inconsistencies and hallucinated content produced by advanced language models. The system uses semantic similarity graphs, attention-based learning, and multi-scorer ensembles to deliver accurate and interpretable hallucination detection. 

We have successfully developed a **Full-Stack Application** featuring a React-based interactive frontend and a robust PyTorch/Flask backend that performs live AI-driven hallucination detection.

## 2. Objective

To design and implement a scalable hallucination detection framework that:
- Detects factually incorrect or unsupported LLM responses with high accuracy.
- Operates effectively in low-supervision and noisy-data environments.
- Uses graph-based learning to model semantic relationships between multiple responses.
- Leverages multi-scorer ensemble systems for robust factuality verification.
- Generates interpretable explanations for each detection to improve user trust.
- Provides a seamless, interactive user interface for real-time hallucination analysis.

## 3. Team & Supervisor

**Department of Computer Science & Engineering**  
Jaypee University of Information Technology, Waknaghat

**Team Members:**
- Arpita Rani – Roll No: 221030055
- Anand Chaudhary – Roll No: 221030123
- Rishal Rana – Roll No: 221030004
- Arnav Sharma – Roll No: 221030059

**Supervisor:**  
Prof. Dr. Vivek Kumar Sehgal

## 4. Key Features & Methodology

We propose a multi-layered hallucination detection framework consisting of the following stages:

- **Input & Response Generation:** Accepts user prompts and generates multiple diverse variations using controlled sampling strategies (powered by TinyLlama).
- **Semantic Embedding Extraction:** Converts responses into high-dimensional vector embeddings using transformer-based models (`S-PubMedBert-MS-MARCO`). Captures contextual and semantic structure of generated text.
- **Graph Construction:** Builds semantic similarity graphs using k-NN and threshold-based edge creation to model relationships between consistent and inconsistent responses.
- **Graph Attention Network (GAT):** Applies attention-based graph neural networks to learn response reliability patterns and classify responses into correct, minor hallucination, moderate hallucination, or hallucinated categories.
- **Explainable AI (XAI) Module / Interactive UI:** A modern React frontend that integrates semantic role labeling (SRL) and attention visualization, highlighting hallucinated spans and providing reasoning-level explanations.

## 5. High-Level Architecture

### Full-Stack Pipeline Overview
- **Frontend (React + Vite):** A responsive, fast UI where users input queries and view live hallucination detection results and confidence scores.
- **Backend API (Flask):** Handles model inference, live embedding generation, dynamic graph construction, and GAT classification.
- **AI Models:** PyTorch Geometric (GAT), SentenceTransformers (Embeddings), and Hugging Face Pipelines (Generative LLM).

### Detection Pipeline
1. **Stage 1 – Fact Extraction & Generation:** User prompt → TinyLlama generates diverse answers (or Semantic Role Labeling generates structured fact tuples).
2. **Stage 2 – Similarity Graph Builder:** Embeddings via `S-PubMedBert` → k-NN Graph → Semantic edges based on cosine similarity thresholds.
3. **Stage 3 – Graph Attention Classifier:** GAT-based hallucination classification model predicts the hallucination state and outputs probabilities.

## 6. Dataset

The system generates its own dataset using controlled prompt-response generation. The dataset includes:
- Correct responses
- Partially correct responses
- Hallucinated responses

Stored in:
- `data/processed/` (.csv, .json)
- `data/raw/` (raw LLM outputs)

## 7. Preprocessing Pipeline

The following preprocessing steps are applied:
- Response normalization and cleaning
- Tokenization and sentence segmentation
- Embedding generation using transformer encoders
- Semantic similarity computation

## 8. Model Components

- **Embedding Module:** Transformer-based encoders for semantic representation.
- **Graph Module:** k-NN based semantic similarity graph construction.
- **Classification Module:** Graph Attention Network (GAT) for hallucination prediction.
- **Scoring Module:** White-box uncertainty scorer, SRL-based factual scorer, Ensemble fusion layer.
- **Explainability Layer:** Highlighted hallucinated spans and attention-based interpretability visualization.

## 9. Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js & npm (for the frontend)
- PyTorch & CUDA-enabled GPU (recommended)
- Virtualenv / Conda

### Backend Setup (Flask API)
The backend hosts the PyTorch models and exposes the `/detect` and `/health` APIs.

```bash
# 1. Navigate to the project root and create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Flask server
cd backend
python app.py
```
*Note: The backend will download the TinyLlama and SentenceTransformer models on the first run. The server runs on `http://localhost:5001`.*

### Frontend Setup (React App)
The frontend provides the interactive user interface for hallucination detection.

```bash
# 1. Open a new terminal and navigate to the frontend directory
cd frontend

# 2. Install Node.js dependencies
npm install

# 3. Start the Vite development server
npm run dev
```
*The React application will be available at `http://localhost:5173` (or the port specified by Vite).*

## 10. Future Work
- Integration of real-time web-based fact verification APIs.
- White-box uncertainty modeling using token entropy.
- Cross-lingual hallucination detection.
- Multimodal hallucination detection (text-image-video).
- Browser extension for real-time hallucination highlighting.

## 11. Acknowledgments
- JUIT CSE Department
- HuggingFace Model Hub
- PyTorch Geometric Community
- Open-source LLM research community

## 12. Video Demonstration
A live demonstration of our project, showcasing the data generation, graph construction, training, and evaluation processes, can be viewed here:

[**Watch the Video Demonstration**](https://drive.google.com/file/d/1Gq3V_0q7QTH5RJBFS0X_nNW3Ckz_oRPG/view?usp=sharing)

## 13. Research Papers

For a complete list of the literature and research papers reviewed for this project, please see the [Literature Review & References](REFERENCES.md) document.
