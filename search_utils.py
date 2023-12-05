import json
import requests

# Function related to google search (via serper) and scrape (via browserless)
def google_search(search_keyword, serper_api_key):    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": search_keyword
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print("google_search for: ",search_keyword,response.text)
    return response.text