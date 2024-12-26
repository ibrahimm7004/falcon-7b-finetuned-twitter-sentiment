from flask import Flask, render_template, request, redirect, url_for
# Import get_tweets and ask_finbert functions
from utils import get_tweets, ask_finbert
from datetime import datetime

app = Flask(__name__)

# To store tweets and sentiment results
tweets = []  # Holds the raw tweet data
sentiment_results = []  # Holds the sentiment analysis results for Screen 4


@app.route('/')
def home():
    return redirect(url_for('screen1'))


@app.route('/screen1')
def screen1():
    return render_template('screen1.html')


@app.route('/screen2', methods=['GET', 'POST'])
def screen2():
    if request.method == 'POST':
        # Get the user input and redirect to screen3 with the query as a parameter
        user_input = request.form['query']
        return redirect(url_for('screen3', query=user_input))
    return render_template('screen2.html')


@app.route('/screen3')
def screen3():
    global tweets
    query = request.args.get('query', '')

    if not query:
        # Redirect back to screen2 if no query is present
        return redirect(url_for('screen2'))

    # Fetch tweets using the query
    raw_data = get_tweets(query)
    tweets = [
        {
            "Tweet Date": datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S +0000 %Y").strftime("%Y-%m-%d %H:%M:%S"),
            "Tweet Text": tweet["text"]
        }
        for tweet in raw_data["tweets"]
    ]

    # Render screen3 with the tweets data
    return render_template('screen3.html', tweets=tweets, query=query)


@app.route('/screen4', methods=['GET'])
def screen4():
    global tweets, sentiment_results

    if not sentiment_results:  # Perform sentiment analysis only once
        sentiment_results = [
            {
                "Tweet Text": tweet["Tweet Text"],
                "Sentiment": ask_finbert(tweet["Tweet Text"])
            }
            for tweet in tweets
        ]

    # Render screen4 with sentiment results
    return render_template('screen4.html', sentiments=sentiment_results)


if __name__ == '__main__':
    app.run(debug=True)
