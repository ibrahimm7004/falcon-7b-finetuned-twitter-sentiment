import requests
API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": "Bearer xxx"}


def scrape_x(topic):
    url = "https://twitter-api45.p.rapidapi.com/search.php"

    querystring = {"query": topic, "search_type": "Latest"}

    headers = {
        "x-rapidapi-key": "ed87c89ff5msh4d695896654ef69p1373afjsn76c8f0ca904f",
        "x-rapidapi-host": "twitter-api45.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    return response.json()


def get_tweets(query):
    # Fetch the raw JSON response using scrape_x
    raw_data = scrape_x(query)

    # Check if the response status is "ok" and timeline data exists
    if raw_data.get('status') == 'ok' and 'timeline' in raw_data:
        # Filter out only the `created_at` and `text` fields
        filtered_data = [
            {
                "created_at": tweet.get("created_at"),
                "text": tweet.get("text")
            }
            for tweet in raw_data['timeline']
            if tweet.get("type") == "tweet"  # Ensure it's a tweet
        ]

        # Return the filtered data as a JSON object
        return {"tweets": filtered_data}
    else:
        # Return an empty result if the status is not ok or timeline is missing
        return {"tweets": []}


def ask_finbert(tweet):
    response = requests.post(API_URL, headers=headers, json={"inputs": tweet})
    x = response.json()
    return x[0][0]['label']
