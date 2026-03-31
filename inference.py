import pandas as pd

def run_pipeline(file_path):
    print("STARTING PROGRAM")

    df = pd.read_csv(file_path)
    original_df = df.copy()

    print("\nInitial Dataset:")
    print(df.head())

    # -------------------------------
    # AUTO DECISION LOGIC
    # -------------------------------
    print("\n🧠 Auto-detecting issues...")

    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✔ Removed {duplicates} duplicate rows")

    # Fill missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.fillna(df.mean(numeric_only=True))
        print(f"✔ Filled {missing} missing values")

    # -------------------------------
    # OUTLIER DETECTION (NEW 🔥)
    # -------------------------------
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = len(df)

        if before != after:
            print(f"✔ Removed outliers from {col}")

    # -------------------------------
    # NORMALIZATION
    # -------------------------------
    if len(numeric_cols) > 0:
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (
            df[numeric_cols].max() - df[numeric_cols].min()
        )
        print("✔ Normalized numeric columns")

    print("\nFinal Cleaned Dataset:")
    print(df.head())

    print("\nPROGRAM FINISHED")

    return original_df, df


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/easy.csv"
    run_pipeline(file_path)