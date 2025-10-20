from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/form", tags=["Form"])

class FormData(BaseModel):
    name: str

@router.post("/submit")
def submit_form(data: FormData):
    return {"message": f"Hello mello, {data.name} bae!"}
