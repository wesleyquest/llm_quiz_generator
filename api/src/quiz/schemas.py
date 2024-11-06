from pydantic import BaseModel


# Shared properties
class Quiz(BaseModel):
    results : str  #[quiz,answer]
    answer : list

class Translate_Quiz(BaseModel):
    results : str


