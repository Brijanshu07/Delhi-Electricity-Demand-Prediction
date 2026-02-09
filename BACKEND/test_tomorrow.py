
import urllib.request
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

API_KEY = os.environ.get("TOMORROW_API_KEY")
LAT = "28.6128"
LON = "77.2311"

def test_api():
    if not API_KEY:
        print("Error: TOMORROW_API_KEY not found in environment.")
        return

    url = f"https://api.tomorrow.io/v4/weather/forecast?location={LAT},{LON}&apikey={API_KEY}"
    print(f"Fetching from: {url}")

    try:
        req = urllib.request.Request(url)
        req.add_header('accept', 'application/json')
        req.add_header('accept-encoding', 'deflate, gzip, br')
        
        with urllib.request.urlopen(req) as response:
            if response.info().get('Content-Encoding') == 'gzip':
                import gzip
                data = gzip.decompress(response.read())
            elif response.info().get('Content-Encoding') == 'br':
                 # python standard lib doesn't support br check if simple read works
                 # usually urllib decodes if 'accept-encoding' is not sent or handled transparently?
                 # actually urllib.request doesn't auto-decode. 
                 # Given the prompt's curl example uses --compressed, we should be careful.
                 # Let's try simple read first, if encoded, we might need simple json.
                 data = response.read()
            else:
                 data = response.read()

            try:
                json_data = json.loads(data)
                print("Successfully fetched data.")
                print("Keys in root:", json_data.keys())
                if "timelines" in json_data:
                    print("Timelines keys:", json_data["timelines"].keys())
                    if "hourly" in json_data["timelines"]:
                        print(f"Hourly forecast count: {len(json_data['timelines']['hourly'])}")
                        print("First hourly entry:", json_data["timelines"]["hourly"][0])
            except json.JSONDecodeError:
                 print("Failed to decode JSON. Raw data might be compressed.")
                 print(data[:100])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_api()
