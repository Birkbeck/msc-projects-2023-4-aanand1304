"""
Created on Jan 21 01:20:38 2023

This script counts for each country the number of tweets about wars and conflicts related to that country.
The output is the monthly count of these tweets for each country in the period between July 2011 and December 2022.

Output file: ACA.csv
"""

# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
from csv import writer
import os
from dotenv import load_dotenv

load_dotenv()

#Place your Twitter API's bearer token here!
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAP6TvQEAAAAASJOSiO6RznUf7ia54NDWshLgwmI%3DJhAnH2c7oIBM46C9gZy9lSXdPGVLl91db2I8vhp5aqZKC5PwHT' 


def create_url(keyword, start_date, end_date, max_results=10):
    search_url = "https://api.twitter.com/2/tweets/search/recent"  # Use 'recent' instead of 'all'
    query_params = {
        'query': keyword,
        'max_results': max_results,
        'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
        'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
        'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
        'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
        'next_token': {}
    }
    return search_url, query_params



def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def is_leap_year(year):
    """Determine whether a year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


months=['01','02','03','04','05','06','07','08','09','10','11','12']
countries_s=['(USA OR America)',
           '(UK OR British OR United Kingdom OR Britain)',
           '(CANADA OR CANADIAN)',
           '(AUSTRALIA)',
           '(Ukraine)',
           '(RUSSIA)',
           '(FRANCE OR FRENCH)',
           '(GERMAN)',
           '(Brazil)',
           '(China OR chinese)',
           '(Japan)',
           '(Pakistan)',
           '(North Korea)',
           '(South Korea)',
           '(India)',
           '(Taiwan)',
           '(NetherLands OR Holland OR Dutch)',
           '(SPAIN OR Spanish)',
           '(Sweden OR Swedish)',
           '(Mexic)',
           '(IRAN)',
           '(ISRAEL)',
           '(Saudi)',
           '(Syria)',
           '(Finland OR FINNISH)',
           '(IRELAND OR IRISH)',
           '(AUSTRIA)',
           '(NORWAY OR Norwegian)',
           '(Switzerland OR swiss)',
           '(ITALY OR ITALIAN)',
           '(MALAYSIA)',
           '(EGYPT)',
           '(TURKEY OR TURKISH)',
           '(portugal OR portuguese)',
           '(Palestin OR West Bank OR GAZA)',
           '(UAE OR United Arab Emirates OR emarat)']

countries=['US','GB','CA','AU','UA','RU','FR','DE','BR','CN','JP','PK',
           'KP','KR','IN','TW','NL','ES','SE','MX','IR','IL','SA','SY',
           'FI','IE','AT','NO','CH','IT','MY','EG','TR','PT','PS','AE']



header=['Date']
for country in countries:
    header.append('WAR/CONFLICT '+country)

conflicts=[header]
# Setting the maximum total tweets to be pulled per month
MAX_TWEETS = 1500
MAX_TWEETS_PER_COUNTRY = int(MAX_TWEETS / len(countries))  # ~42 tweets per country

with open('ACA.csv', 'a', newline="") as f:
    for year in range(2023, 2025):  # Adjust years to 2023-2024
        for month in months:
            if int(month) > 4 and year == 2024:
                break

            conflict = [month + '/' + str(year)]
            total_tweets_pulled = 0

            for c_index, country in enumerate(countries):
                if total_tweets_pulled >= MAX_TWEETS:
                    break

                bearer_token = auth()
                headers = create_headers(bearer_token)
                keyword = f"({countries_s[c_index]} WAR MILITARY) OR ({countries_s[c_index]} WAR ARMED FORCE) OR ({countries_s[c_index]} CONFLICT POLITIC) OR ({countries_s[c_index]} MILITARY ATTACK) OR ({countries_s[c_index]} ARMED FORCE ATTACK) lang:en"
                start_time = f"{year}-{month}-01T00:00:00.000Z"
                end_time = f"{year}-{month}-28T23:59:59.000Z"  # Adjust for days in the month
                max_results = min(400, MAX_TWEETS_PER_COUNTRY)

                count = 0
                next_token = None
                flag = True

                while flag and count < MAX_TWEETS_PER_COUNTRY:
                    url = create_url(keyword, start_time, end_time, max_results)
                    json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
                    result_count = json_response['meta']['result_count']

                    if 'next_token' in json_response['meta']:
                        next_token = json_response['meta']['next_token']
                        count += result_count
                        total_tweets_pulled += result_count
                        time.sleep(6)
                    else:
                        count += result_count
                        total_tweets_pulled += result_count
                        flag = False
                    time.sleep(5)

                conflict.append(count)
            writer_object = writer(f)
            writer_object.writerow(conflict)
            f.flush()
f.close()






