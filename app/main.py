from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.services.drug_lookup import init_store, lookup_drug

app = FastAPI(title="MediAssist Drug Lookup")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_store()

@app.get("/lookup")
def lookup(drug: str = Query(..., min_length=1)):
    return lookup_drug(drug)
