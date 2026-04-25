import requests

API_URL = "http://localhost:8000/chat"

def chat(message: str) -> str:
    response = requests.post(API_URL, json={"message": message})
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    print("Football Analyst Console (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        try:
            reply = chat(user_input)
            print(f"\n{reply}\n")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect. Is the server running? (python main.py)\n")
        except requests.exceptions.HTTPError as e:
            print(f"Error: {e}\n")
