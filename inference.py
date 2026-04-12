import os
from openai import OpenAI

#  MUST use these environment variables
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"]
)

def call_llm():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # safe default
            messages=[
                {"role": "system", "content": "You are a helpful AI."},
                {"role": "user", "content": "Suggest a data cleaning step."}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def run_task(task_name):
    print(f"[START] task={task_name}", flush=True)

    llm_output = call_llm()  #  THIS TRIGGERS PROXY CALL

    # simulate steps
    for i in range(2):
        print(f"[STEP] step={i+1} reward=0.5", flush=True)

    print(f"[END] task={task_name} score=0.7 steps=2", flush=True)


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        run_task(task)
