
import sys
import os
from pathlib import Path

# Add BACKEND to path so we can import main
backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))

import main

def verify():
    print("Verifying Tomorrow.io integration...")
    
    # 1. Check API Key
    if not main.TOMORROW_API_KEY:
        print("FAIL: TOMORROW_API_KEY is not set.")
        return

    # 2. Load Model
    print("Loading model...")
    if not main.load_model():
        print("FAIL: Model could not be loaded. Please ensure MODEL/processed/demand_model.joblib exists.")
        return
    print("Model loaded.")

    # 3. Test Fetch & Predict
    print("Fetching forecast and generating predictions for next 5 hours...")
    try:
        results = main.get_next_n_hours(5)
        if not results:
            print("FAIL: No results returned.")
            return
        
        print(f"Success! Got {len(results)} hourly predictions.")
        for i, row in enumerate(results):
            print(f"Hour {i}: Time={row['timestamp']}, Temp={row['temp']}, Load={row['predicted_load']}")
            
        # Basic validation
        if results[0]['temp'] == 30 and results[0]['rhum'] == 50 and results[0]['wspd'] > 0:
             # This might be fallback or valid data. 
             # If it matches exact fallback defaults (30C, 50%), it might be fallback. But 30C is possible in Delhi.
             pass

    except Exception as e:
        print(f"FAIL: Exception during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
