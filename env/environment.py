import pandas as pd

class DataEnv:
    def __init__(self):
        self.df = None
        self.current_task = 0

        #  3 TASKS
        self.tasks = [
            {"name": "easy", "target_duplicates": 1},
            {"name": "medium", "target_duplicates": 1},
            {"name": "hard", "target_duplicates": 2}
        ]

    def reset(self):
        task = self.tasks[self.current_task % 3]
        self.current_task += 1

        # dataset per task
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
            "reward": 0.5,
            "done": False,
            "info": {"task": task["name"]}
        }

    def step(self, action):
        if self.df is None:
            return {
                "observation": None,
                "reward": 0.5,
                "done": False,
                "info": {"error": "Call /reset first"}
            }

        # Apply action
        if action == "remove_duplicates":
            self.df = self.df.drop_duplicates()

        # ---------------- GRADER LOGIC ----------------
        duplicates_remaining = self.df.duplicated().sum()

        if duplicates_remaining == 0:
            #  SUCCESS CONDITION
            reward = 0.9
            done = True
        else:
            #  PARTIAL PROGRESS
            reward = 0.6
            done = False

        return {
            "observation": self.df.to_dict(),
            "reward": float(reward),
            "done": done,
            "info": {
                "rows": len(self.df),
                "duplicates_remaining": int(duplicates_remaining),
                "task": self.task["name"]
            }
        }

    def get_state(self):
        return {
            "data": self.df.to_dict() if self.df is not None else None
        }
