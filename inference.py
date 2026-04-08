import requests
import time

BASE_URL = "https://purandhareswari-k-smart-data-openenv.hf.space"

def run():
    try:
        # Step 1: Reset environment
        res = requests.post(f"{BASE_URL}/reset", timeout=10)
        data = res.json()

        print("Initial State:", data)

        # Step 2: Perform actions
        actions = ["remove_duplicates", "fill_missing", "outlier_clean"]

        for action in actions:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"action": action},
                timeout=10
            )
            result = response.json()
            print(f"Action: {action}", result)

            if result.get("done", False):
                break

        # Step 3: Final state
        state = requests.get(f"{BASE_URL}/state", timeout=10)
        print("Final State:", state.json())

    except Exception as e:
        print({"error": str(e)})


if __name__ == "__main__":
    run()
