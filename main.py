from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from schemas import CategoryGetDto, CategoryResDto
from app import fetchExpenseCategoryUsingEmbeddings

app = FastAPI(title="Basic FastAPI", version="1.0.0")

# CORS (adjust for your frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/category", response_model=CategoryResDto, tags=["category"])
def fetchCategory(
    categoryRequest: CategoryGetDto,
):
    try:
        category, scores = fetchExpenseCategoryUsingEmbeddings(categoryRequest.title)
        return CategoryResDto(category=category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
