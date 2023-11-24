import json

def load_json_data(file: str) -> dict:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def write_json_data(file: str, data: dict):
    try:
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except IOError:
        print(f"Error: Unable to write to file '{file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


