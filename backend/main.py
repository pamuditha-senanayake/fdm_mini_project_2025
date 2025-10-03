# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from components import stockpredict, promotionpredict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(stockpredict.router)
app.include_router(promotionpredict.router)

@app.get("/")
def root():
    return {"message": "RetailIQ backend running."}
