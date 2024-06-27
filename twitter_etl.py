# Run the following in command line

# python -m pip install tweepy (in terminal)
# python -m pip install pandas (in terminal)
# python -m pip install json (in terminal)
# python -m pip install datetime (in terminal)
# python -m pip install s3fs (in terminal)
# python -m pip install nltk (in terminal)

import tweepy
import pandas as pd 
import json
from datetime import datetime
import s3fs 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def run_twitter_etl():

    access_key = "" 
    access_secret = "" 
    consumer_key = ""
    consumer_secret = ""

    # Twitter authentication
    auth = tweepy.OAuthHandler(access_key, access_secret)   
    auth.set_access_token(consumer_key, consumer_secret) 

    # # # Creating an API object 
    api = tweepy.API(auth)
    tweets = api.user_timeline(screen_name='@jtimberlake', 
                            # 200 is the maximum allowed count
                            count=200,
                            include_rts = False,
                            # Necessary to keep full_text 
                            # otherwise only the first 140 words are extracted
                            tweet_mode = 'extended'
                            )

    list = []
    for tweet in tweets:
        text = tweet._json["full_text"]

        refined_tweet = {"user": tweet.user.screen_name,
                        'text' : text,
                        'favorite_count' : tweet.favorite_count,
                        'retweet_count' : tweet.retweet_count,
                        'created_at' : tweet.created_at}
        
        list.append(refined_tweet)

    df = pd.DataFrame(list)

    # Convert 'created_at' to datetime and extract month and year
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['month'] = df['created_at'].dt.month
    df['year'] = df['created_at'].dt.year
    # Concatenate month and year into a single column
    df['month_year'] = df['created_at'].dt.to_period('M').astype(str)

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis
    df['sentiment'] = df['text'].apply(lambda text: sia.polarity_scores(text)['compound'])

    df.to_csv('refined_tweets.csv')
