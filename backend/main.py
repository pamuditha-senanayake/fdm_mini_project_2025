# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from components import stockpredict

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router from stockpredict
app.include_router(stockpredict.router)

@app.get("/")
def root():
    return {"message": "RetailIQ backend is running."}
