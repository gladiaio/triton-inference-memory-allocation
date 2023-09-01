from fastapi import FastAPI
from .ClientSender import ClientSender

app = FastAPI()
client = ClientSender()

@app.get("/")
async def root():
    return await client()
