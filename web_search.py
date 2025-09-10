import os
import requests

def serpapi_search(query, num_results=5, serpapi_key=None):
    serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return []

    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": serpapi_key,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
        return results
    except Exception as e:
        print("SerpAPI error:", e)
        return []
