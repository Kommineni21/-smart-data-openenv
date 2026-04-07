title: Smart Data OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
AI-Powered Smart Data Cleaning Environment
This project is a custom Reinforcement Learning (RL) Environment designed to automate the data cleaning process. It allows AI agents to learn which cleaning actions provide the highest quality improvement for a given dataset.

Key Features
Automated Quality Scoring: Real-time scoring based on missing values and duplicates.
Atomic Action Space: Clean, modular actions for duplicates, missing data, and outliers.
RL-Ready Interface: Fully compatible with the openenv-core library and Pydantic validation.
Action Types
Action	Description
remove_duplicates	Deletes identical rows to ensure data uniqueness.
fill_missing	Imputes numerical null values using column-wise means.
outlier_clean	Uses IQR filtering to remove extreme statistical anomalies.
Scoring & Reward System
The environment uses a weighted penalty system to calculate the data health: Score = 100 × (1 - (0.6 × Missing% + 0.4 × Duplicate%))

The Reward is the direct improvement in the score from the starting baseline. An episode is considered Done once a perfect score of 100 is achieved or the 10-step limit is reached.

Technical Stack
Python / Pandas: Core data manipulation.
Pydantic: Strict data modeling for API stability.
Hugging Face Spaces: Interactive web hosting.
OpenEnv-Core: Standardized RL environment framework.
