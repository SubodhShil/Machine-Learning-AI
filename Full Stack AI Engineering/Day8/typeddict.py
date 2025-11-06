from typing import TypedDict, Optional
from pydantic import BaseModel, Field

class Person(BaseModel):
    key_themes: list[str] = Field(description="")
    name: str
    age: int