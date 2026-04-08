import pandas as pd
import numpy as np
import random
from pydantic import BaseModel
from env.graders import grade_easy, grade_medium, grade_hard


class DataAction(BaseModel):
    action_type: str


class DataObservation(BaseModel):
    data_summary: str
    row_count: int
    score: float
    reward: float = 0.0
    done: bool = False
    episode_id: str = "1"
    step_count: int = 0
    status_message: str = ""


class DataCleaningEnv:
    def __init__(self):
        self.files = ["data/easy.csv", "data/medium.csv", "data/hard.csv"]
        self.max_steps = 10
        self.df = None
        self.current_step = 0
        self.reset()

    @property
    def state(self):
        if self.df is None:
            return "No data loaded. Click Reset."
        return self._get_obs()

    # ✅ CORRECT RESET
    def reset(self) -> DataObservation:
        self.file_path = random.choice(self.files)

        try:
            self.df = pd.read_csv(self.file_path)
        except:
            self.df = pd.DataFrame({
                "age": [25, 25, np.nan, 40, 150],
                "salary": [50000, 50000, 60000, 70000, 999999]
            })

        self.current_step = 0

        return self._get_obs(message=f"Environment Reset. Loaded {self.file_path}")

    # ✅ CORRECT STEP
    def step(self, action: DataAction) -> DataObservation:
        self.current_step += 1
        act = action.action_type

        valid_actions = ["remove_duplicates", "fill_missing", "outlier_clean"]

        if act not in valid_actions:
            return self._get_obs(reward=-20, message="Invalid action")

        prev_df = self.df.copy()

        if act == "remove_duplicates":
            self.df = self.df.drop_duplicates().reset_index(drop=True)

        elif act == "fill_missing":
            for col in self.df.select_dtypes(include=[np.number]).columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        elif act == "outlier_clean":
            for col in self.df.select_dtypes(include=[np.number]).columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[
                    ~((self.df[col] < (Q1 - 1.5 * IQR)) |
                      (self.df[col] > (Q3 + 1.5 * IQR)))
                ]

        reward = self.compute_reward(prev_df, self.df, act)
        reward += self.apply_grader_bonus()

        return self._get_obs(reward=reward, message=f"Executed {act}")

    # -------- OBS --------
    def _get_obs(self, reward=0.0, done=False, message=""):
        score = self.calculate_score(self.df)

        if score >= 100:
            done = True
            message = "Perfect dataset cleaned!"
        elif self.current_step >= self.max_steps:
            done = True
            message = "Max steps reached."

        status = (
            f"Rows:{len(self.df)} | "
            f"Missing:{self.df.isnull().sum().sum()} | "
            f"Duplicates:{self.df.duplicated().sum()} | "
            f"Outliers:{self.count_outliers(self.df)}"
        )

        return DataObservation(
            data_summary=self.df.head(10).to_string(),
            row_count=len(self.df),
            score=score,
            reward=reward,
            done=done,
            episode_id="1",
            step_count=self.current_step,
            status_message=status + " | " + message
        )

    # -------- REWARD --------
    def compute_reward(self, prev_df, new_df, action):
        reward = 0

        reward += (prev_df.isnull().sum().sum() - new_df.isnull().sum().sum()) * 5
        reward += (prev_df.duplicated().sum() - new_df.duplicated().sum()) * 10
        reward += (self.count_outliers(prev_df) - self.count_outliers(new_df)) * 8

        if prev_df.equals(new_df):
            reward -= 15

        return reward

    def apply_grader_bonus(self):
        if "easy" in self.file_path:
            return grade_easy(self.df) * 20
        elif "medium" in self.file_path:
            return grade_medium(self.df) * 20
        else:
            return grade_hard(self.df) * 20

    def calculate_score(self, df):
        total = df.size if df.size > 0 else 1

        missing = df.isnull().sum().sum() / total
        dup = df.duplicated().sum() / len(df)
        out = self.count_outliers(df) / total

        return round(max(0, 100 * (1 - (0.5 * missing + 0.3 * dup + 0.2 * out))), 2)

    def count_outliers(self, df):
        count = 0
        for col in df.select_dtypes(include=[np.number]):
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            count += ((df[col] < Q1 - 1.5 * IQR) |
                      (df[col] > Q3 + 1.5 * IQR)).sum()
        return count

    async def reset_async(self):
        return self.reset()

    async def step_async(self, action: DataAction):
        return self.step(action)
