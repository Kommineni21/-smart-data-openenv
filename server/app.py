from fastapi import FastAPI

app = FastAPI()

# ---- YOUR ORIGINAL LOGIC ----

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {"message": "reset done"}

@app.post("/step")
def step(action: dict):
    return {"message": "step executed", "action": action}

@app.get("/state")
def state():
    return {"state": "running"}

# ---- REQUIRED FOR VALIDATOR ----
def main():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
