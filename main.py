from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from schemas import (
    TextClassificationRequestDto,
    TextClassificationResponseDto,
)
from app import fetchTextLabelUsingBert

app = FastAPI(title="Lafier Text Classification", version="1.0.0")

# CORS (adjust for your frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=TextClassificationResponseDto)
def classifyText(
    textClassificationRequest: TextClassificationRequestDto,
):
    try:
        label = fetchTextLabelUsingBert(
            textClassificationRequest.text, textClassificationRequest.labels
        )
        return TextClassificationResponseDto(label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
