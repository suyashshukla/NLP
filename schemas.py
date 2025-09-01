from pydantic import BaseModel, Field

class CategoryGetDto(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)

class CategoryResDto(BaseModel):
    category: str