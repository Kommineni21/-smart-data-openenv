import pandas as pd
from typing import Dict, Any
from pydantic import BaseModel


# ----------------------------
# MODELS
# ----------------------------
class DataAction(BaseModel):
    action: str


class DataObservation(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ----------------------------
# ENVIRONMENT
# ----------------------------
class DataEnv:

    def __init__(self):
        self.df = None
        self.current_task = "easy"
        self.tasks = ["easy", "medium", "hard"]

        #  REQUIRED by OpenEnv
        self.step_count = 0


    # ----------------------------
    # RESET
    # ----------------------------
    def reset(self):
        self.step_count = 0  #  reset counter

        if self.current_task == "medium":
            data = {
                "name": ["A", "B", "C", "A", "D"],
                "age": [20, 25, None, 20, 40]
            }
        elif self.current_task == "hard":
            data = {
                "name": ["A", "B", "C", "A", "E", "F"],
                "age": [20, None, 30, 20, 1000, 25]
            }
        else:  # easy
            data = {
                "name": ["A", "B", "C", "A"],
                "age": [20, 25, 30, 20]
            }

        self.df = pd.DataFrame(data)

        return DataObservation(
            observation=self.df.to_dict(),
            reward=0.5,
            done=False,
            info={"task": self.current_task}
        )


    # ----------------------------
    # STEP
    # ----------------------------
    def step(self, action: DataAction):

        #  REQUIRED
        self.step_count += 1

        #  safety fix
        if self.df is None:
            self.reset()

        initial_rows = len(self.df)

        # ---------------- ACTIONS ----------------
        if action.action == "remove_duplicates":
            self.df = self.df.drop_duplicates()

        elif action.action == "fill_missing":
            self.df = self.df.fillna(self.df.mean(numeric_only=True))

        elif action.action == "outlier_clean":
            for col in self.df.select_dtypes(include=["number"]).columns:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                self.df = self.df[
                    (self.df[col] >= q1 - 1.5 * iqr) &
                    (self.df[col] <= q3 + 1.5 * iqr)
                ]

        # ---------------- REWARD ----------------
        final_rows = len(self.df)

        if final_rows < initial_rows:
            reward = 0.8
        elif final_rows == initial_rows:
            reward = 0.6
        else:
            reward = 0.4

        # STRICT RANGE (Phase 2 requirement)
        reward = min(max(reward, 0.1), 0.95)

        done = reward >= 0.8 or self.step_count >= 10

        return DataObservation(
            observation=self.df.to_dict(),
            reward=reward,
            done=done,
            info={
                "task": self.current_task,
                "steps": self.step_count
            }
        )


    # ----------------------------
    # STATE
    # ----------------------------
    def state(self):
        return {
            "data": self.df.to_dict() if self.df is not None else {},
            "task": self.current_task,
            "steps": self.step_count
        }
