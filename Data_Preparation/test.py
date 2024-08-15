import requests

# Your API key and the DOI
api_key = 'your_api_key_here'
doi = '10.1109/JIOT.2022.3210197'  # The DOI of your document

# Construct the URL for the DOI
url = f'https://api.elsevier.com/content/article/doi/{doi}?APIKey={api_key}&httpAccept=text/plain'

# Send the request
response = requests.get(url)

if response.status_code == 200:
    print("Full Text Content (first 500 characters):")
    print(response.text[:500])
elif response.status_code == 403:
    print("Access forbidden. You may not have the necessary access rights.")
elif response.status_code == 404:
    print("Document not found. The DOI might be incorrect or not available via this API.")
else:
    print(f"An error occurred. Status Code: {response.status_code}")
