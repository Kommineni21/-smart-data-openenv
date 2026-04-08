from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import DataEnv

app = FastAPI()

env = DataEnv()

class Action(BaseModel):
    action: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    return env.step(action.action)

@app.get("/state")
def state():
    return env.get_state()
