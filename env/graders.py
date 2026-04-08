def grade_easy(df):
    return 1.0 if df.duplicated().sum() == 0 else 0.0


def grade_medium(df):
    no_duplicates = df.duplicated().sum() == 0
    no_missing = df.isnull().sum().sum() == 0

    if no_duplicates and no_missing:
        return 1.0
    elif no_duplicates or no_missing:
        return 0.5
    return 0.0


def grade_hard(df):
    no_duplicates = df.duplicated().sum() == 0
    no_missing = df.isnull().sum().sum() == 0

    numeric_df = df.select_dtypes(include=["number"])
    normalized = True

    if not numeric_df.empty:
        if numeric_df.max().max() > 1 or numeric_df.min().min() < 0:
            normalized = False

    score = 0.0
    if no_duplicates:
        score += 0.3
    if no_missing:
        score += 0.3
    if normalized:
        score += 0.4

    return score
