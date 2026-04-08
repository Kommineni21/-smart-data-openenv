import pandas as pd

class DataEnv:
    def __init__(self):
        self.df = None
        self.current_task = 0

        #  3 TASKS (required)
        self.tasks = [
            {"name": "easy", "data": [["A", 20], ["B", 25], ["A", 20]]},
            {"name": "medium", "data": [["A", 20], ["B", 25], ["C", 30], ["A", 20]]},
            {"name": "hard", "data": [["A", 20], ["B", 25], ["C", 30], ["A", 20], ["B", 25]]}
        ]

    def reset(self):
        # Rotate tasks
        task = self.tasks[self.current_task % 3]
        self.current_task += 1

        self.df = pd.DataFrame(task["data"], columns=["name", "age"])

        return {
            "observation": self.df.to_dict(),
            "reward": 0.5,  #  not 0
            "done": False,
            "info": {"task": task["name"]}
        }

    def step(self, action):
        # Safety check
        if self.df is None:
            return {
                "observation": None,
                "reward": 0.5,
                "done": False,
                "info": {"error": "Call /reset first"}
            }

        if action == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)

            #  reward logic (strictly between 0 and 1)
            if after < before:
                reward = 0.9   # success (NOT 1.0)
                done = True
            else:
                reward = 0.6   # partial
                done = False

        elif action == "fill_missing":
            reward = 0.4
            done = False

        elif action == "outlier_clean":
            reward = 0.3
            done = False

        else:
            reward = 0.2
            done = False

        return {
            "observation": self.df.to_dict(),
            "reward": float(reward),
            "done": done,
            "info": {"rows": len(self.df)}
        }

    def get_state(self):
        return {
            "data": self.df.to_dict() if self.df is not None else None
        }
