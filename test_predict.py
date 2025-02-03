import requests

url = "http://localhost:9696/predict"

client = {
    'EMA7': 20768.484481399686, 'EMA14': 20490.05642871806, 'EMA30': 20146.747057307526, 
    'ROC7': 2.1014717561301666, 'ROC14': 8.17196844974428, 'ROC30': 7.776440929635841, 
    'MOM7': 290.88281199999983, 'MOM14': 1359.4785149999989, 'MOM30': 1379.6367189999983, 
    'RSI7': 60.814408187394406, 'RSI14': 59.67152906192006, 'RSI30': 53.807929837547135, 
    '%K7': 61.75344612013603, '%D7': 81.3790872452849, '%K14': 76.77367219169297, 
    '%D14': 88.90518384754644, '%K30': 83.35818148868839, '%D30': 91.98348494419609
}

try:
    response = requests.post(url, json=client)
    response.raise_for_status()  # Check for HTTP errors
    
    # Parse the JSON response
    result = response.json()
    
    # Print the result
    print("Prediction Response:", result)
    
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")  # Example: 404 Not Found or 500 Internal Server Error
except requests.exceptions.RequestException as req_err:
    print(f"Request error occurred: {req_err}")  # For any other requests-related errors
except ValueError as json_err:
    print(f"Error parsing JSON: {json_err}")  # If the response isn't JSON
