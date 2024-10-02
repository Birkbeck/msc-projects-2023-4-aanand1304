from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
import sys
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd
from webdriver_manager.firefox import GeckoDriverManager
import concurrent.futures
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

## Load configuration
con_file = open("config.json")  # Place your Elsevier key in this file
config = json.load(con_file)
con_file.close()

## Initialise client
client = ElsClient(config['apikey'])

# this list will be used in the query of Elsevier (attack list)
attack_list = [
    'Ransomware',
    'Deepfake',
    'IoT Device Attack',
    'WannaCry AND (Ransomware OR Attack)',
    'Adversarial Attack',
    '"Data Poisoning"',
    'Cryptojack*',
]

# this list will be used to count the number of mentions of each attack type by the harvester (attack keywords list)
mention_list = [
    ('Ransomware',),
    ('Deepfake',),
    ('IoT Device', 'IoT Attack', 'Attack on IoT', 'Attack the IoT'),
    ('Cryptojack',),
    ('WannaCry',),
    ('Adversarial Attack',),
    ('Data Poisoning',)
]

semaphore = threading.Semaphore(3)  # Limit to 3 browser windows

def create_browser_instance():
    options = Options()
    driver = webdriver.Firefox(options=options, service=Service(GeckoDriverManager().install()))
    return driver

def extract_url_from_scopus(attack, month, year):
    slm = []
    print("Searching for ", attack)
    query = f"TITLE({attack}) OR ABS({attack}) OR KEY({attack}) AND PUBDATETXT({month} {year})"
    
    # Retry logic for handling rate limiting
    for attempt in range(5):  # Retry up to 5 times
        try:
            doc_srch = ElsSearch(query, 'scopus')
            doc_srch.execute(client, get_all=True)
            if 'Result set was empty' in str(doc_srch.results):
                return slm
            else:
                for col in doc_srch.results:
                    slm.append(col)
            break
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):  # Handle rate limit error
                print(f"Rate limit exceeded. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e  # Re-raise the exception if it's not a 429 error
    return slm

def process_url(result, mention, driver):
    semaphore.acquire()  # Acquire a semaphore before using a browser
    totalQ = 0
    
    try:
        url = result['link'][2]['@href']
        print("URL:", url)
        try:
            driver.get(url)
        except:
            time.sleep(20)
            driver.get(url)
        time.sleep(1)
        delay = 600  # seconds
        try:
            WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'Highlight-module__akO5D')))
        except TimeoutException:
            print("Loading took too much time!")
            return 0
        
        # Click index keywords
        keyword_elements = driver.find_elements(By.TAG_NAME, "button")
        keyword_element = None
        while True:
            failed = False
            for element in keyword_elements:
                try:
                    if 'Indexed keywords' in element.text:
                        keyword_element = element
                        print("Keyword element found!")
                        break
                except:
                    print("Button issue")
                    keyword_element = None
                    failed = True
                    time.sleep(1)
                    break
            if not failed:
                break
            else:
                keyword_elements = driver.find_elements(By.TAG_NAME, "button")

        if keyword_element is not None:
            keyword_element.click()
            time.sleep(1)
        
        # Counting mentions
        while True:
            failed = False
            text_elements = driver.find_elements(By.CLASS_NAME, "Highlight-module__akO5D")
            for element in text_elements:
                try:
                    element_text = element.text
                except:
                    failed = True
                    break
                count = sum(element_text.replace("-", " ").upper().count(x.upper().replace("-", " ")) for x in mention)
                print("Count:", count)
                totalQ += count
            if not failed:
                break
        
    finally:
        semaphore.release()  # Release the semaphore

    return totalQ

if __name__ == "__main__":
    # Lists of years and months for the search
    years_list = list(range(2023, 2024))
    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    search_results_list = []

    # Create and maintain a pool of 3 browser instances
    browser_pool = [create_browser_instance() for _ in range(3)]
    
    # Log in to the site manually using these browsers before continuing
    input("Please log in to the site using the opened browser windows, then press Enter to continue...")

    # Part 1 - Collect the URLs of all relevant documents (for each month and each attack type)
    for year in years_list:
        print("Searching in year:", str(year))
        
        for month in month_list:
            if year == 2023 and month in month_list[4:]:
                continue

            print("Searching in year/month:", str(year), "/", month)
            with ThreadPoolExecutor(max_workers=3) as executor:  # Reduce concurrency
                slsm = list(executor.map(extract_url_from_scopus, attack_list, repeat(month), repeat(year)))
            search_results_list.append(slsm)

    # Part 2 - Harvesting and counting the number of mentions of each attack type during each month
    print("-------------------------- Harvesting Time ---------------------------")
    data = []

    for index, row in enumerate(search_results_list):
        print("Out of", len(search_results_list), "Harvesting for column", index)
        attack_map = {mention[0]: 0 for mention in mention_list}

        for j, col in enumerate(row):
            with ProcessPoolExecutor() as executor:
                # Reuse the same browsers from the pool
                results = list(executor.map(process_url, col, [mention_list[j]] * len(col), [browser_pool[i % 3] for i in range(len(col))]))
                attack_map[mention_list[j][0]] += sum(results)
            
            print(f"For attack {mention_list[j]} sum is now:", attack_map[mention_list[j][0]])

        data_list = []
        for key in attack_map:
            print(key, ": ", attack_map[key])
            data_list.append(attack_map[key])
        data.append
