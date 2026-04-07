import pandas as pd

def process_dataset(file_path, level):
    print(f"\n===== RUNNING {level.upper()} DATASET =====\n")

    df = pd.read_csv(file_path)
    original_df = df.copy()

    print("Initial Dataset:")
    print(df.head())
    print(f"\nShape: {df.shape}")

    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"\nRemoved {duplicates} duplicate rows")
    else:
        print("\nNo duplicate rows found")

    # Fill missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.fillna(df.mean(numeric_only=True))
        print(f"Filled {missing} missing values")
    else:
        print("No missing values found")

    # Normalize numeric columns
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        df[num_cols] = (df[num_cols] - df[num_cols].min()) / (
            df[num_cols].max() - df[num_cols].min()
        )
        print("Normalized numeric columns")

    # Outlier detection (IQR)
    print("\nOutliers detected:")
    outliers = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        count = ((df[col] < (Q1 - 1.5 * IQR)) |
                 (df[col] > (Q3 + 1.5 * IQR))).sum()

        outliers[col] = count
        print(f"{col}: {count}")

    # Dataset Quality Score
    total_cells = original_df.shape[0] * original_df.shape[1]

    missing_penalty = original_df.isnull().sum().sum() / total_cells
    duplicate_penalty = original_df.duplicated().sum() / original_df.shape[0]
    outlier_penalty = sum(outliers.values()) / total_cells

    quality_score = 100 - (
        (missing_penalty * 40) +
        (duplicate_penalty * 30) +
        (outlier_penalty * 30)
    )

    quality_score = max(0, round(quality_score, 2))

    print(f"\nDataset Quality Score: {quality_score}/100")

    print("\nFinal Dataset Preview:")
    print(df.head())

    print("\n===== DONE =====\n")


def run_all():
    process_dataset("data/easy.csv", "easy")
    process_dataset("data/medium.csv", "medium")
    process_dataset("data/hard.csv", "hard")


if __name__ == "__main__":
    print(" STARTING PROGRAM")
    run_all()
    print("PROGRAM FINISHED")
