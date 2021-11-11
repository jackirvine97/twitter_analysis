import requests

url = "https://api.twitter.com/2/tweets/search/all?query="

payload={}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)



