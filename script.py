from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def main():
    return {"message": "Hello World"}

@app.get('/{name}')
def hello_name(name: str):
    return {"message": f"Hello {name}"}