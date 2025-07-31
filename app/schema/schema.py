from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class QuestionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]