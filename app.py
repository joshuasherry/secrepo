from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/predict_link")
def predict_link(source_id: str, target_id: str):
    # Map source and target to node indices
    u = node_id_to_idx[source_id]
    v = node_id_to_idx[target_id]
    features = compute_features(u, v, g, embeddings, current_time=pd.Timestamp.now())
    score = clf.predict_proba([features])[0][1]
    return {"source": source_id, "target": target_id, "link_probability": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
