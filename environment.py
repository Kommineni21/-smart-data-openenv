import pandas as pd

class DataEnv:
    def __init__(self):
        self.df = None

    def reset(self):
        # Sample dataset with duplicates
        data = {
            "name": ["A", "B", "C", "A"],
            "age": [20, 25, 30, 20]
        }

        self.df = pd.DataFrame(data)

        return {
            "observation": self.df.to_dict(),
            "reward": 0.0,
            "done": False,
            "info": {"task": "medium"}
        }

    def step(self, action):
        # ✅ SAFETY FIX (prevents 500 error)
        if self.df is None:
            return {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {
                    "error": "Call /reset before /step"
                }
            }

        # ✅ MAIN LOGIC
        if action == "remove_duplicates":
            self.df = self.df.drop_duplicates()

            # ✅ FINAL CORRECT CONDITION
            if self.df.duplicated().sum() == 0:
                reward = 1.0
                done = True
            else:
                reward = 0.5
                done = False
        else:
            reward = -0.1
            done = False

        return {
            "observation": self.df.to_dict(),
            "reward": float(reward),
            "done": done,
            "info": {
                "rows": len(self.df)
            }
        }

    def get_state(self):
        return {
            "data": self.df.to_dict() if self.df is not None else None
        }
