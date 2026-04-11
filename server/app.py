from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

def main():
    # Run your original app.py (DO NOT import it)
    subprocess.run([
        "uvicorn",
        "app:app",
        "--host", "0.0.0.0",
        "--port", "7860"
    ])

if __name__ == "__main__":
    main()
