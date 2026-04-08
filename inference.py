import requests
import os
from openai import OpenAI

BASE_URL = "https://purandhareswari-k-smart-data-openenv.hf.space"

def run():
    task_name = "smart_data_cleaning"
    step_count = 0
    total_reward = 0

    try:
        # ---------------- LLM CALL (MANDATORY) ----------------
        try:
            client = OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"]
            )

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Plan data cleaning steps"}
                ],
                max_tokens=10
            )

            print("[LLM] call_success", flush=True)

        except Exception as e:
            print(f"[LLM_ERROR] {str(e)}", flush=True)

        # ---------------- START ----------------
        res = requests.post(f"{BASE_URL}/reset", timeout=10)
        data = res.json()

        print(f"[START] task={task_name}", flush=True)

        actions = ["remove_duplicates", "fill_missing", "outlier_clean"]

        # ---------------- STEPS ----------------
        for i, action in enumerate(actions, start=1):
            response = requests.post(
                f"{BASE_URL}/step",
                json={"action": action},
                timeout=10
            )
            result = response.json()

            reward = float(result.get("reward", 0))
            done = result.get("done", False)

            total_reward += reward
            step_count = i

            print(f"[STEP] step={i} reward={reward}", flush=True)

            if done:
                break

        # ---------------- END ----------------
        score = total_reward
        print(f"[END] task={task_name} score={score} steps={step_count}", flush=True)

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)


if __name__ == "__main__":
    run()
