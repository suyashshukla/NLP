from pydantic import BaseModel, Field


class TextClassificationRequestDto(BaseModel):
    text: str = Field(..., min_length=1, max_length=100)
    labels: list[str] = Field(..., min_items=1)


class TextClassificationResponseDto(BaseModel):
    label: str
