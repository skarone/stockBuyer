import urllib.parse
import json
import datetime
import random
import os
import pickle
from datetime import timedelta
import oauthlib
from requests_oauthlib import OAuth1Session

import operator
from textblob import TextBlob
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer

class GetTweets:
    def parse_config(self):
      config = {}
      # from file args
      if os.path.exists('data/config.json'):
          with open('data/config.json') as f:
              config.update(json.load(f))
      # should have something now
      return config

    def oauth_req(self, url, http_method="GET", post_body=None,
                  http_headers=None):
      config = self.parse_config()
      twitter = OAuth1Session(config.get('consumer_key'),client_secret=config.get('consumer_secret'), resource_owner_key=config.get('access_token'),resource_owner_secret=config.get('access_token_secret'))
      resp = twitter.request(http_method,url)
      return resp
    
    #start getTwitterData
    def getData(self, keyword, params = {}):
        maxTweets = 1000
        url = 'https://api.twitter.com/1.1/search/tweets.json?'    
        data = {'q': keyword, 'lang': 'en', 'result_type': 'recent', 'count': maxTweets, 'include_entities': 0}
        #Add if additional params are passed
        if params:
            for key, value in params.items():
                data[key] = value
        
        url += urllib.parse.urlencode(data)
        response = self.oauth_req(url)          
        jsonData = json.loads(response.content.decode('UTF-8'))

        if 'errors' in jsonData:
            print("API Error")
            print(jsonData['errors'])
        else:
            return jsonData
    #end    

    def getTwitterData(self, symbol, day=datetime.datetime.now()):
        tomorrow = day+timedelta(days=+1)
        params = {'since': day.strftime("%Y-%m-%d"),
            'until': tomorrow.strftime("%Y-%m-%d")
        }
        jsonData = self.getData(symbol, params)
        tweets = []
        for item in jsonData['statuses']:
           myData = {}
           myData["created_at"] = item["created_at"]
           myData["text"] = item["text"]
           myData["user"] = item["user"]["screen_name"]
           tweets.append(myData)
        return tweets

    def getSentiment(self, symbol, day=datetime.datetime.now()):
        data = self.getTwitterData(symbol, day)
        sorted_keys = sorted(data, key=lambda kv: kv["created_at"])
        final_sentiment = 0
        positive = 0
        negative = 0
        neutral = 0
        for tweet in sorted_keys:
            textB = TextBlob(tweet["text"])
            sentiment = textB.sentiment.polarity
            print(tweet["created_at"], sentiment, tweet["text"])
            if sentiment <0.00:
                negative += 1
            elif sentiment >0.00:
                positive += 1
            else:
                neutral += 1
            final_sentiment += sentiment
        print("Positive:", positive)
        print("Negative:", negative)
        print("Neutral:", neutral)
        print("Final:", final_sentiment)
        return positive - negative

 
 
if __name__ == "__main__":
    a  = GetTweets()
    day = datetime.datetime.now() - timedelta(days=3)
    print(a.getSentiment('$NWS', day), day)
