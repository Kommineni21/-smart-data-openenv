import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

def process(file):
    try:
        df = pd.read_csv(file.name)

        original_df = df.copy()

        output = " STARTING DATA CLEANING\n\n"

        # Initial info
        output += f" Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n"

        # ========================
        # 1. Remove duplicates
        # ========================
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            output += f" Removed {duplicates} duplicate rows\n"
        else:
            output += " No duplicate rows\n"

        # ========================
        # 2. Missing values
        # ========================
        missing_before = df.isnull().sum()

        if missing_before.sum() > 0:
            df = df.fillna(df.mean(numeric_only=True))
            output += f" Filled missing values\n"
        else:
            output += "No missing values\n"

        # Missing values graph
        plt.figure()
        missing_before.plot(kind='bar')
        plt.title("Missing Values per Column")
        plt.xticks(rotation=45)
        missing_plot = plt.gcf()
        plt.close()

        # ========================
        # 3. Normalize
        # ========================
        num_cols = df.select_dtypes(include=['number']).columns

        if len(num_cols) > 0:
            df[num_cols] = (df[num_cols] - df[num_cols].min()) / (
                df[num_cols].max() - df[num_cols].min()
            )
            output += " Normalized numeric columns\n"

        # ========================
        # 4. Outlier Detection
        # ========================
        outliers = {}
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers[col] = ((df[col] < (Q1 - 1.5 * IQR)) |
                             (df[col] > (Q3 + 1.5 * IQR))).sum()

        output += "\n Outliers detected per column:\n"
        for k, v in outliers.items():
            output += f"{k}: {v}\n"

        # ========================
        # Save cleaned file
        # ========================
        output_file = "cleaned_output.csv"
        df.to_csv(output_file, index=False)

        # ========================
        # Explanation
        # ========================
        explanation = """
 Steps Applied:
1. Removed duplicate rows
2. Filled missing values using mean
3. Normalized numeric columns
4. Detected outliers using IQR method
"""

        return output, df, missing_plot, explanation, output_file

    except Exception as e:
        return f" ERROR: {str(e)}", None, None, None, None


# ========================
# UI DESIGN
# ========================
with gr.Blocks() as demo:
    gr.Markdown("#  Smart Data Cleaning AI")
    gr.Markdown("Upload any dataset → Clean → Analyze → Visualize → Download")

    file_input = gr.File(label=" Upload CSV")

    run_btn = gr.Button(" Run Pipeline")

    output_text = gr.Textbox(label=" Pipeline Output", lines=12)

    output_df = gr.Dataframe(label=" Cleaned Dataset")

    plot = gr.Plot(label="Missing Values Chart")

    explanation = gr.Textbox(label=" Explanation")

    download = gr.File(label=" Download Cleaned CSV")

    run_btn.click(
        fn=process,
        inputs=file_input,
        outputs=[output_text, output_df, plot, explanation, download]
    )

demo.launch()
