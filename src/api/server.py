import uvicorn
from fastapi import FastAPI
from src.config import API_HOST, API_PORT, API_DEBUG


app = FastAPI()

@app.get("/health")
def healthcheck():
    """
    Check if the API is running.
    """

    return {"status": "API is running"}


if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host=API_HOST, port=API_PORT, reload=API_DEBUG)

