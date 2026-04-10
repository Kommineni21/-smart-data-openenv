import pandas as pd

class DataEnv:
    def __init__(self):
        self.df = None
        self.current_task = 0

        #  REQUIRED TASK STRUCTURE
        self.tasks = [
            {
                "name": "easy",
                "grader": {"type": "scalar", "min": 0.1, "max": 0.95}
            },
            {
                "name": "medium",
                "grader": {"type": "scalar", "min": 0.1, "max": 0.95}
            },
            {
                "name": "hard",
                "grader": {"type": "scalar", "min": 0.1, "max": 0.95}
            }
        ]

    def reset(self):
        task = self.tasks[self.current_task % 3]
        self.current_task += 1

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

            #  CRITICAL: expose tasks WITH graders
            "tasks": self.tasks,

            "info": {
                "task": task["name"]
            }
        }

    def step(self, action):
        if self.df is None:
            return {
                "observation": None,
                "reward": 0.5,
                "done": False,
                "tasks": self.tasks,
                "info": {"error": "Call /reset first"}
            }

        if action == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)

            if after < before:
                reward = 0.9   #  valid range
                done = True
            else:
                reward = 0.6
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

            #  ALWAYS include tasks
            "tasks": self.tasks,

            "info": {
                "task": self.task["name"],
                "rows": len(self.df),
                "duplicates_remaining": int(self.df.duplicated().sum())
            }
        }

    def get_state(self):
        return {
            "data": self.df.to_dict() if self.df is not None else None,
            "task": self.task["name"] if hasattr(self, "task") else None,
            "tasks": self.tasks
        }
