from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config_loader import load_config
from src.pipeline import run_pipeline

app = FastAPI(
    title="Vector — Lead Intelligence API",
    description=(
        "Local-only service wrapping the Vector pipeline for reuse "
        "outside the Streamlit demo. Not deployed to Hugging Face."
    ),
    version="1.0.0",
)


class RunAnalysisRequest(BaseModel):
    
    data_mode: str = "demo"


@app.get("/health")
def health() -> dict:
    
    return {"status": "ok"}


@app.post("/run-analysis")
def run_analysis(request: RunAnalysisRequest) -> dict:
   
    config = load_config()
    try:
        output = run_pipeline(config, data_mode=request.data_mode)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{exc} Valid values are 'demo' or 'local'.",
        )

    result_df = output["result"]
    return {
        "n_leads": output["n_leads"],
        "model_auc": output["model_auc"],
        "silhouette_avg": output["silhouette_avg"],
        "leads": result_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
   
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
