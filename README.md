**Explainable Graph-Based Hallucination Detection for Large Language Models**

Rapid, accurate, and interpretable detection of hallucinated content in Large Language Model (LLM) outputs using graph learning, semantic analysis, and ensemble scoring.

**1. Project Overview**

This repository contains the implementation of our Major Project:

Explainable Graph-Based Hallucination Detection for Large Language Models (LLMs)

The project focuses on identifying factual inconsistencies and hallucinated content produced by advanced language models. The system uses semantic similarity graphs, attention-based learning, and multi-scorer ensembles to deliver accurate and interpretable hallucination detection.

**2. Objective**

To design and implement a scalable hallucination detection framework that:

Detects factually incorrect or unsupported LLM responses with high accuracy.

Operates effectively in low-supervision and noisy-data environments.

Uses graph-based learning to model semantic relationships between multiple responses.

Leverages multi-scorer ensemble systems for robust factuality verification.

Generates interpretable explanations for each detection to improve user trust.

**3. Team
**
Department of Computer Science & Engineering
Jaypee University of Information Technology, Waknaghat

Arpita Rani – Roll No: 221030055

Anand Chaudhary – Roll No: 221030123

Rishal Rana – Roll No: 221030004

Arnav Sharma – Roll No: 221030059

**Supervisor:**
Prof. Dr. Vivek Kumar Sehgal

**4. Key Features & Methodology**

We propose a multi-layered hallucination detection framework consisting of the following stages:

**Input & Response Generation**

Accepts user prompts and LLM-generated responses.

Generates multiple diverse variations using controlled sampling strategies.

**Semantic Embedding Extraction**

Converts responses into high-dimensional vector embeddings using transformer-based models.

Captures contextual and semantic structure of generated text.

**Graph Construction**

Builds semantic similarity graphs using k-NN and threshold-based edge creation.

Models relationships between consistent and inconsistent responses.

**Graph Attention Network (GAT)**

Applies attention-based graph neural networks to learn response reliability patterns.

Classifies responses into correct, partially correct, or hallucinated categories.

**Explainable AI (XAI) Module**

Integrates semantic role labeling (SRL) and attention visualization.

Highlights hallucinated spans and provides reasoning-level explanations.

**5. High-Level Architecture**
**Pipeline Overview**

**Input**
LLM-generated text
Optional reference knowledge sources

Stage 1 – Fact Extraction Module
Semantic Role Labeling (SRL) + structured fact tuple generation

Stage 2 – Similarity Graph Builder
Embedding → k-NN Graph → Semantic edges

Stage 3 – Graph Attention Classifier
GAT-based hallucination classification model
**
Output**

Hallucination probability score

Highlighted factual inconsistencies

Explanation heatmaps

**6. Dataset**

The system generates its own dataset using controlled prompt-response generation.

The dataset includes:

Correct responses

Partially correct responses

Hallucinated responses

Stored in:

data/processed/ (.csv, .json)

data/raw/ (raw LLM outputs)
**
7. Preprocessing Pipeline**

The following preprocessing steps are applied:

Response normalization and cleaning

Tokenization and sentence segmentation

Embedding generation using transformer encoders

Semantic similarity computation

**8. Model Components**
Embedding Module

Transformer-based encoders for semantic representation.

Graph Module

k-NN based semantic similarity graph construction.

Classification Module

Graph Attention Network (GAT) for hallucination prediction.

Scoring Module

Black-box scorer

White-box uncertainty scorer

SRL-based factual scorer

Ensemble fusion layer

Explainability Layer

Highlighted hallucinated spans

Attention-based interpretability visualization

**9. Installation & Setup**
Prerequisites

Python 3.8+

PyTorch

CUDA-enabled GPU (recommended)

Virtualenv / Conda

Setup Steps
# Clone repository
git clone https://github.com/your-username/AI-Hallucination-Detector.git
cd AI-Hallucination-Detector

# Create virtual environment (optional)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Run the Application
python main.py

**10. Future Work**

Integration of real-time web-based fact verification APIs

White-box uncertainty modeling using token entropy

Cross-lingual hallucination detection

Multimodal hallucination detection (text-image-video)

Browser extension for real-time hallucination highlighting

**11. Acknowledgments**

JUIT CSE Department

HuggingFace Model Hub

PyTorch Geometric Community

Open-source LLM research community


**Research Papers:**

Arpita Rani: 1) [UQLM](https://arxiv.org/pdf/2507.06196v1): A Python Package for Uncertainty Quantification in
Large Language Models.
2) https://arxiv.org/abs/2311.16479
3) https://arxiv.org/abs/2509.04664
4) https://www.ijsr.net/getabstract.php?paperid=SR241229170309
5) https://arxiv.org/abs/2311.05232


Rishal Rana: 1)[RAG-HAT](https://aclanthology.org/2024.emnlp-industry.113.pdf): A Hallucination-Aware Tuning Pipeline for LLM in
Retrieval-Augmented Generation  


Anand Chaudhary: 1)[Lookback Lens](https://arxiv.org/pdf/2407.07071): Detecting and Mitigating Contextual Hallucinations in
Large Language Models Using Only Attention Maps.     
                 2)A [Comprehensive Survey of Hallucination](https://arxiv.org/pdf/2405.09589) in Large Language, Image, Video and Audio Foundation Models


Arnav Sharma: 1)[A Survey on Hallucination in Large Language Models](https://arxiv.org/pdf/2311.05232):
 Principles, Taxonomy, Challenges, and Open Questions.




 


