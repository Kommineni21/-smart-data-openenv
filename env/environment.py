import pandas as pd

class DataEnv:
    def __init__(self):
        self.df = None
        self.current_task = 0

        #  3 TASKS (explicit)
        self.tasks = [
            {"name": "easy"},
            {"name": "medium"},
            {"name": "hard"}
        ]

    def reset(self):
        task = self.tasks[self.current_task % 3]
        self.current_task += 1

        # Dataset per task
        if task["name"] == "easy":
            data = [["A", 20], ["B", 25], ["A", 20]]
        elif task["name"] == "medium":
            data = [["A", 20], ["B", 25], ["C", 30], ["A", 20]]
        else:
            data = [["A", 20], ["B", 25], ["C", 30], ["A", 20], ["B", 25]]

        self.df = pd.DataFrame(data, columns=["name", "age"])
        self.task = task

        return {
            "observation": self.df.to_dict(),
            "reward": 0.5,  # must be between (0,1)
            "done": False,

            #  CRITICAL: expose tasks + graders
            "tasks": [
                {"name": "easy", "grader": "reward > 0.8"},
                {"name": "medium", "grader": "reward > 0.8"},
                {"name": "hard", "grader": "reward > 0.8"}
            ],

            "info": {"task": task["name"]}
        }

    def step(self, action):
        if self.df is None:
            return {
                "observation": None,
                "reward": 0.5,
                "done": False,
                "task": None,
                "info": {"error": "Call /reset first"}
            }

        # Apply action
        if action == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)

            #  reward strictly between (0,1)
            if after < before:
                reward = 0.9   # success (NOT 1.0)
                done = True
            else:
                reward = 0.6   # partial progress
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

            #  expose current task
            "task": self.task["name"],

            "info": {
                "rows": len(self.df),
                "duplicates_remaining": int(self.df.duplicated().sum())
            }
        }

    def get_state(self):
        return {
            "data": self.df.to_dict() if self.df is not None else None,
            "task": self.task["name"] if hasattr(self, "task") else None
        }
