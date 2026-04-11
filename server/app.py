from fastapi import FastAPI
from env.environment import DataEnv, DataAction

app = FastAPI()

env = DataEnv()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    return env.reset()


@app.post("/step")
def step(action: DataAction):
    return env.step(action)


@app.get("/state")
def state():
    return env.state()
