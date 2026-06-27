import ee

print("Launching Google Earth Engine Authentication...")
print("A browser window will open. Please log in, grant permissions, and copy the authorization code.")
print("Paste the code below when prompted.")

try:
    ee.Authenticate()
    print("Authentication Complete! The token is saved.")
except Exception as e:
    print(f"Authentication failed: {e}")
