import requests
import pandas as pd

# API URL
url = "https://iot.zolnoi.app/api/v1/energy/"

# Query parameters
params = {
    "type": "harmonics",
    "duration_type": "custom",
    "response_type": "raw",
    "start_time": "2025-08-01T01:30:00Z",
    "end_time": "2025-08-01T01:45:00Z",
    "machine_id": "257"
}

# Headers
headers = {
    "accept": "application/json",
    "api_key": "mqdm_CgwNaRsb62Ziy5ePw"
}

# Fetch data
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()   # Parse JSON response

    # Convert to DataFrame (assuming response is list of dicts)
    df = pd.DataFrame(data)

    # Save to CSV
    output_file = "energy_data.csv"
    df.to_csv(output_file, index=False)

    print(f"Data saved to {output_file}, {len(df)} rows.")
else:
    print(f"Request failed: {response.status_code}")
    print(response.text)
