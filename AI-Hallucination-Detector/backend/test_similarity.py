import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

print("Loading Embedding Model...")
embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', device='cpu')

sentences = [
    "The recommended treatment for SARS-CoV-2 includes antiviral medications like Nirmatrelvir/Ritonavir and supportive care.",
    "The standard treatment for severe acute respiratory syndrome coronavirus 2 is prescribing antivirals combined with clinical supportive care.",
    "Patients with SARS-CoV-2 are typically treated using antiviral drugs and supportive clinical management.",
    "The primary treatment protocol for COVID-19 involves administering antivirals along with necessary supportive care in a clinical setting.",
    "Antiviral treatments, alongside close supportive clinical care, are the widely accepted medical interventions for SARS-CoV-2."
]

print("Computing Embeddings...")
embeddings = embedder.encode(sentences, convert_to_tensor=True)

print("Computing Cosine Similarity Matrix...")
# Compute pairwise cosine similarity
similarities = []
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
        similarities.append(sim.item())
        print(f"Similarity {i} vs {j}: {sim.item():.4f}")

avg_sim = sum(similarities) / len(similarities)
min_sim = min(similarities)
print(f"\nAverage Similarity: {avg_sim:.4f}")
print(f"Minimum Similarity: {min_sim:.4f}")

if min_sim >= 0.85:
    print("\nSUCCESS: All sentences exceed the 0.85 threshold! These will form a dense graph.")
else:
    print("\nFAILURE: Some sentences failed the 0.85 threshold. These will break the graph.")
