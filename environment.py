import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Dict, Any, Optional

# 1. Models - All required attributes for the interface
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

# 2. The Environment Class
class DataCleaningEnv:
    def __init__(self, file_path="data/Hard.csv", task_name="Data Cleaning"):
        self.file_path = file_path
        self.task_name = task_name
        self.df = None
        self.initial_score = 0.0
        self.current_step = 0
        self.max_steps = 10 

    @property
    def state(self):
        if self.df is None:
            return "No data loaded. Click Reset."
        return self._get_obs()

    def calculate_score(self, df):
        if df.empty: return 0.0
        missing_pct = df.isnull().sum().sum() / (df.size if df.size > 0 else 1)
        dup_pct = df.duplicated().sum() / (len(df) if len(df) > 0 else 1)
        # Scoring logic: 60% weight on missing values, 40% on duplicates
        score = 100 * (1 - (0.6 * missing_pct + 0.4 * dup_pct))
        return round(max(0, score), 2)

    def reset(self) -> DataObservation:
        try:
            self.df = pd.read_csv(self.file_path)
        except:
            # Fallback data for demonstration
            self.df = pd.DataFrame({
                "age": [25, 25, np.nan, 40, 150], 
                "salary": [50000, 50000, 60000, 70000, 80000]
            })
        
        self.initial_score = self.calculate_score(self.df)
        self.current_step = 0
        return self._get_obs(message="Environment Reset. Target: Score 100.")

    def _get_obs(self, reward: float = 0.0, done: bool = False, message: str = "") -> DataObservation:
        current_score = self.calculate_score(self.df)
        
        # Smart termination logic
        if current_score >= 100.0:
            done = True
            message = "Goal Reached! Data is now perfectly clean."
        elif self.current_step >= self.max_steps:
            done = True
            message = "Max steps reached."

        return DataObservation(
            data_summary=self.df.head(10).to_string(),
            row_count=len(self.df),
            score=current_score,
            reward=reward,
            done=done,
            episode_id="1",
            step_count=self.current_step,
            status_message=message
        )

    def step(self, action: DataAction) -> DataObservation:
        self.current_step += 1
        act = action.action_type
        message = f"Executed {act}"
        
        if act == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            message = f"Success: Removed {before - len(self.df)} duplicate rows."
            
        elif act == "fill_missing":
            nulls = self.df.isnull().sum().sum()
            for col in self.df.select_dtypes(include=[np.number]).columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            message = f"Success: Filled {nulls} missing values."
            
        elif act == "outlier_clean":
            before = len(self.df)
            for col in self.df.select_dtypes(include=[np.number]).columns:
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[~((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR)))]
            message = f"Success: Removed {before - len(self.df)} outlier rows."
        
        current_score = self.calculate_score(self.df)
        reward = float(current_score - self.initial_score)
        
        return self._get_obs(reward=reward, message=message)

    async def reset_async(self):
        return self.reset()

    async def step_async(self, action: DataAction):
        return self.step(action)
