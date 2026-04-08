import requests

BASE_URL = "https://purandhareswari-k-smart-data-openenv.hf.space"

def run():
    print("[START] task=data-clean env=smart-data model=baseline")

    # Reset
    res = requests.post(f"{BASE_URL}/reset")
    data = res.json()

    # Step
    action = {"action": "remove_duplicates"}
    res = requests.post(f"{BASE_URL}/step", json=action)
    result = res.json()

    reward = result["reward"]
    done = result["done"]

    print(f"[STEP] step=1 action=remove_duplicates reward={reward:.2f} done={str(done).lower()} error=null")

    success = done and reward == 1.0

    print(f"[END] success={str(success).lower()} steps=1 score={reward:.2f} rewards={reward:.2f}")

if __name__ == "__main__":
    run()
