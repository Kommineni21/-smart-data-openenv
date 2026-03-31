def grade_easy(df):
    return 1.0 if df.duplicated().sum() == 0 else 0.0


def grade_medium(df):
    score = 0

    if df.duplicated().sum() == 0:
        score += 0.5

    if df.isnull().sum().sum() == 0:
        score += 0.5

    return score


def grade_hard(df):
    score = 0

    if df.duplicated().sum() == 0:
        score += 0.25

    if df.isnull().sum().sum() == 0:
        score += 0.25

    numeric_df = df.select_dtypes(include=['number'])

    if not numeric_df.empty and numeric_df.max().max() <= 1:
        score += 0.25

    if "salary" in df.columns and df["salary"].dtype != object:
        score += 0.25

    return score