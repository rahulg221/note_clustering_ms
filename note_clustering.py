from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics import silhouette_score
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

app = FastAPI()

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Request body that takes in a list of strings
class RequestBody(BaseModel):
    notes: list[str]
    
@app.get("/")
def main():
    return {"message": "Hello"}

@app.get("/label")
async def cluster_notes(request_body: RequestBody):
    notes = request_body.notes

    if not notes:
        return {"error": "No notes provided"}
    
    # Encode the notes
    embeddings = model.encode(notes)

    # Compute silhouette scores
    silhouette_scores = []
    K_range = range(2, 10) 

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)

    # Find the optimal number of clusters
    optimal_k = K_range[np.argmax(silhouette_scores)]

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    # Compile results in a dataframe
    df = pd.DataFrame({"Note": notes, "Label": labels})
    df_sorted = df.sort_values(by="Label")

    # Convert DataFrame to JSON format
    json_result = df_sorted.to_dict(orient="records")

    return {"optimal_k": optimal_k, "clusters": json_result}

