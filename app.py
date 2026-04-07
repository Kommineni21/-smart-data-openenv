import uvicorn
from openenv.core.env_server import create_web_interface_app
from env.environment import DataCleaningEnv, DataAction, DataObservation

app = create_web_interface_app(
    env=lambda: DataCleaningEnv(),   # ✅ FIXED (no file_path)
    action_cls=DataAction,
    observation_cls=DataObservation
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
