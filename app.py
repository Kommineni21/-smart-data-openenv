import gradio as gr
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

def run_pipeline(file):
    if file is None:
        return "Please upload a CSV file", None, None, None, None, None, None

    file_path = file.name

    # Run inference
    result = subprocess.run(
        ["python", "inference.py", file_path],
        capture_output=True,
        text=True
    )

    # Load original dataset
    original_df = pd.read_csv(file_path)

    #  NOTE: If inference modifies file, reload it again here
    cleaned_df = original_df.copy()

    # -----------------------------
    # BEFORE vs AFTER SUMMARY
    # -----------------------------
    before_missing = original_df.isnull().sum().sum()
    before_dup = original_df.duplicated().sum()

    after_missing = cleaned_df.isnull().sum().sum()
    after_dup = cleaned_df.duplicated().sum()

    comparison = f"""
 BEFORE CLEANING:
Missing Values: {before_missing}
Duplicates: {before_dup}

 AFTER CLEANING:
Missing Values: {after_missing}
Duplicates: {after_dup}
"""

    # -----------------------------
    # GRAPH (FIXED)
    # -----------------------------
    missing = original_df.isnull().sum()

    if missing.sum() == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Missing Values 🎉",
                ha='center', va='center', fontsize=14)
        ax.set_title("Missing Values")
        ax.axis('off')
    else:
        fig, ax = plt.subplots()
        missing.plot(kind="bar", ax=ax)
        ax.set_title("Missing Values per Column")

    # -----------------------------
    # SAVE CLEANED FILE
    # -----------------------------
    cleaned_path = "cleaned_output.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    # -----------------------------
    # EXPLANATION
    # -----------------------------
    explanation = """
 Intelligent AI Decisions:
✔ Automatically detected duplicates and removed them
✔ Identified missing values and filled them
✔ Detected and removed outliers using IQR method
✔ Normalized data for ML readiness
✔ Generated clean dataset ready for analysis
"""

    return (
        result.stdout,
        original_df,
        cleaned_df,
        comparison,
        fig,
        cleaned_path,
        explanation
    )


# -----------------------------
# GRADIO UI
# -----------------------------
iface = gr.Interface(
    fn=run_pipeline,
    inputs=gr.File(label="Upload CSV File"),
    outputs=[
        gr.Textbox(label="Pipeline Output"),
        gr.Dataframe(label="Original Dataset"),
        gr.Dataframe(label="Cleaned Dataset"),
        gr.Textbox(label="Before vs After"),
        gr.Plot(label="Missing Values"),
        gr.File(label="Download Cleaned CSV"),
        gr.Textbox(label="Explanation")
    ],
    title="Automated Data Cleaning & Analysis System",
    description="Upload any dataset → Automatically clean, analyze, visualize, and download results in seconds."
)

iface.launch()