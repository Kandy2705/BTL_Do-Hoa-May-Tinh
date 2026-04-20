import requests

API_KEY = "rf_PDP3D9g3HhTnh1gPOQzQGuVsHIU2"

# List workspaces
resp = requests.get(
    f"https://api.roboflow.com/v1/workspace?api_key={API_KEY}"
)

print(f"Status: {resp.status_code}")
print(resp.text[:2000])