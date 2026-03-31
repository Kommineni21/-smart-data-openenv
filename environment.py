import pandas as pd
from env.models import Observation

class DataCleaningEnv:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def reset(self):
        self.df = pd.read_csv(self.file_path)
        return self._get_observation()

    def state(self):
        return self.df

    def step(self, action):
        reward = 0
        action_type = action.get("action_type")

        if action_type == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)
            reward += 0.3 if before != after else -0.1

        elif action_type == "fill_missing":
            self.df = self.df.fillna(0)
            reward += 0.3

        elif action_type == "clean_salary":
            if "salary" in self.df.columns:
                self.df["salary"] = (
                    self.df["salary"]
                    .astype(str)
                    .str.replace("k", "000")
                    .str.replace("M", "000000")
                    .str.replace(",", "")
                )
                self.df["salary"] = pd.to_numeric(self.df["salary"], errors="coerce")
                reward += 0.4

        elif action_type == "normalize":
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            self.df[numeric_cols] = (
                (self.df[numeric_cols] - self.df[numeric_cols].min()) /
                (self.df[numeric_cols].max() - self.df[numeric_cols].min())
            )
            reward += 0.4

        elif action_type == "explain":
            reward += 0.1

        done = (
            self.df.isnull().sum().sum() == 0 and
            self.df.duplicated().sum() == 0
        )

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return Observation(
            data_preview=str(self.df.head()),
            missing_values=self.df.isnull().sum().sum(),
            duplicates=self.df.duplicated().sum()
        )