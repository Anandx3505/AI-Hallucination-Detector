import os
import sys
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch.nn.functional as F
from torch_geometric.data import Data
import time

# Add graph directory to path so we can import the GAT model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'graph')))
from GAT import GAT

app = Flask(__name__)
CORS(app)

# Global model state
models = {
    "gat": None,
    "embedder": None,
    "encoder": None, # The sentence transformer
    "llm_pipeline": None, # The generative LLM
    "ready": False
}

# Absolute path to weights
WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights', 'GAT_379.pt'))

def init_models():
    """Load models once at module startup."""
    print("Loading models... this may take a moment.")
    try:
        # 1. Load the 768-d Sentence Transformer for node feature generation
        models["encoder"] = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
        # 1.5 Load Generative LLM
        print("Loading TinyLlama generative model (slow on cold boot)...")
        models["llm_pipeline"] = pipeline(
            'text-generation', 
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            device_map="auto"
        )
        
        # 2. Reconstruct the GAT Architecture
        # The GAT was trained on 768-d inputs, embedded to 128-d via SimCLR MLP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_channels = 768
        hidden_channels = 32
        out_channels = 3 # Original graph.y.shape[1] is 3
        
        embedder_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels), 
            torch.nn.ReLU(), 
            torch.nn.Linear(in_channels, 128)
        )
        gat_model = GAT(embedder_mlp, n_in=in_channels, hid=hidden_channels, n_classes=out_channels, dropout=0.0)
        
        # 3. Load State Dict with compatibility patch
        print(f"Loading weights from {WEIGHTS_PATH}")
        # Explicitly use weights_only=False to bypass PyTorch 2.6 security restriction on torch_geometric.data.Data if it somehow bled in
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint) # Extract nested state_dict if present
        
        if "conv.lin_src.weight" in state_dict:
            state_dict["conv.lin.weight"] = state_dict.pop("conv.lin_src.weight")
            state_dict.pop("conv.lin_dst.weight", None)
            
        gat_model.load_state_dict(state_dict)
        gat_model.eval()
        gat_model.to(device)
        
        models["gat"] = gat_model
        models["ready"] = True
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        models["ready"] = False


# Helper: Decode output to 4-class ordinal based on rewrite_labels
def decode_prediction(logit_vector):
    """
    Simulates utils_graph.rewrite_labels for a single node's logit vector.
    """
    round_out = logit_vector.sigmoid().round()
    if round_out[-1] == 1.:
        return "correct"
    elif round_out[-2] == 1.:
        return "minor_hallucination"
    elif round_out[-3] == 1.:
        return "moderate_hallucination"
    else:
        return "hallucinated"


@app.route('/health', methods=['GET'])
def health_check():
    """Frontend polling endpoint."""
    if models["ready"]:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "loading"}), 503


@app.route('/detect', methods=['POST'])
def detect():
    """Live Detection Endpoint."""
    if not models["ready"]:
        return jsonify({"error": "Models not loaded yet"}), 503
        
    data = request.json
    query = data.get('query', '')
    use_real_llm = data.get('use_real_llm', False)
    
    if not query:
        return jsonify({"error": "Query required"}), 400
        
    try:
        answers = []
        
        if use_real_llm:
            # REAL LLM INFERENCE
            prompt_template = f"<|system|>\nYou are a medical AI. Provide 5 slightly different, distinct, but factually consistent 1-sentence answers to the user's question. Number them 1 to 5.</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"
            
            generation = models["llm_pipeline"](
                prompt_template, 
                max_new_tokens=256, 
                temperature=0.8, 
                do_sample=True,
                return_full_text=False
            )
            raw_text = generation[0]['generated_text']
            
            import re
            parsed_answers = re.findall(r'^\s*\d+[\.\)]\s*(.+)', raw_text, re.MULTILINE)
            
            if len(parsed_answers) >= 5:
                answers = parsed_answers[:5]
            else:
                print(f"LLM Parsing missed 5 targets. Falling back. Found:\n{raw_text}")
                # Fallback to fragmented if instruction-following fails entirely
                answers = [
                    f"{query} The standard clinical approach involves localized therapy followed by systemic observation.",
                    f"{query} Some studies suggest using experimental compound X-99, though it is not yet FDA approved.",
                    f"{query} Typically, patients are advised to immediately cease all physical activity and undergo surgery.",
                    f"{query} The first-line medical intervention is a combination of beta-blockers and immediate rest.",
                    f"{query} Alternative medicine practices recommend herbal supplements and crystal therapy for this condition."
                ]
        else:
            # FAST DEMO MODE
            if "treatment" in query.lower() or "sars" in query.lower():
                answers = [
                    "The recommended treatment for SARS-CoV-2 includes antiviral medications like Nirmatrelvir/Ritonavir and supportive care.",
                    "The standard treatment for severe acute respiratory syndrome coronavirus 2 is prescribing antivirals combined with clinical supportive care.",
                    "Patients with SARS-CoV-2 are typically treated using antiviral drugs and supportive clinical management.",
                    "The primary treatment protocol for COVID-19 involves administering antivirals along with necessary supportive care in a clinical setting.",
                    "Antiviral treatments, alongside close supportive clinical care, are the widely accepted medical interventions for SARS-CoV-2."
                ]
            else:
                answers = [
                    f"{query} The standard clinical approach involves localized therapy followed by systemic observation.",
                    f"{query} Some studies suggest using experimental compound X-99, though it is not yet FDA approved.",
                    f"{query} Typically, patients are advised to immediately cease all physical activity and undergo surgery.",
                    f"{query} The first-line medical intervention is a combination of beta-blockers and immediate rest.",
                    f"{query} Alternative medicine practices recommend herbal supplements and crystal therapy for this condition."
                ]
        
        # Live Embedding Step
        device = next(models["gat"].parameters()).device
        embeddings = models["encoder"].encode(answers, convert_to_tensor=True).to(device)
        
        # Live Graph Construction (k-NN)
        num_nodes = embeddings.shape[0]
        distances = torch.zeros((num_nodes, num_nodes), device=device)
        for i in range(num_nodes):
            distances[i] = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings, dim=-1)
            
        # Mask self-connections
        mask = torch.triu(torch.ones((num_nodes, num_nodes), device=device), diagonal=0).bool()
        distances[mask] = -1.0
        
        # Create edges using strict threshold matching make_graph.py
        threshold = 0.85
        edge_indices = []
        edge_attrs = []
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes): # Upper triangle
                if distances[i, j] >= threshold:
                    # Add undirected edge
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    val = distances[i, j].item()
                    edge_attrs.append(val)
                    edge_attrs.append(val)
                    
        # Convert to tensors
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float, device=device).unsqueeze(1)
        else:
            # Fallback trivial graph if nothing connects
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float, device=device)

        # Live GAT Classification
        with torch.no_grad():
            out_logits = models["gat"](embeddings, edge_index, edge_attr=edge_attr)
            
        # Decode Output per node
        results = []
        confidences = []
        for i in range(num_nodes):
            logits = out_logits[i]
            # Proxy confidence: mean of max sigmoids
            confidence = logits.sigmoid().max().item() * 100
            confidences.append(confidence)
            
            status = decode_prediction(logits)
            
            results.append({
                "id": i,
                "text": answers[i],
                "status": status,
                "confidence": round(confidence, 1)
            })
            
        
        # Optionally, simulate the "Graph Confidence" as the average confidence of the nodes
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0
            
        return jsonify({
            "results": results,
            "graph_confidence": round(overall_confidence, 1)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Start background model loading
init_models()

if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0', debug=False, use_reloader=False, threaded=True)

