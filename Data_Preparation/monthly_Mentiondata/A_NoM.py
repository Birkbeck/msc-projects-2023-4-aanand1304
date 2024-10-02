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
from selenium.webdriver.chrome.service import Service  # <-- CHANGED
from selenium.webdriver.chrome.options import Options  # <-- CHANGED
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager  # <-- CHANGED

   
## Load configuration
con_file = open("config.json")#Place your Elsevier key in this file
config = json.load(con_file)
con_file.close()

## Initialise client
client = ElsClient(config['apikey'])

#harvester config (used to count the number of mentions)
options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)  # <-- CHANGED



# this list will be used in the query of Elsevier (attack list)
attack_list=['"Denial-of-Service Attack" OR {DDoS}',
'Phishing',
'Ransomware',
'Password Attack',
'"SQL Injection"',
'("Account Hijack*") OR ("Account Takeover")',
'Website Deface*',
'Trojan Attack', 
'Vulnerabilit* Attack exploit*', 
'"Zero-day" AND Attack',
'"Advanced Persistent Threat"',
'"cross-site scripting"',
'Malware OR TROJAN OR SPYWARE OR MALVERTIS* OR RANSOMWARE OR ("COMPUTER VIRUS") OR (WORM ATTACK) OR (KEYLOG* Attack) OR (KEYSTROKE LOG Attack) OR (MALICIOUS CODE) OR (MALICIOUS SOFTWARE) OR ADWARE OR ROOTKIT OR BOTNET OR (BACKDOOR ATTACK)',
'"Data Breach" OR "Data Leak*" OR "information leak*" OR "Data SPILL" OR "unintentional information disclos*"', 
'(Disinformation spread) OR (Misinformation spread) OR ("false information" spread) OR ("misleading information" spread) OR Deepfake',
'"Targeted Attack"',
'Adware',
'"Brute Force" Attack',
'Malvertis*',
'Backdoor Attack',
'Botnet',
'Cryptojack*',
'Worm Attack',
'Spyware',
'"Man-in-the-middle" OR {MITM}',
'"DNS Spoof*" OR "DNS cache poison*"',
'Pegasus Spyware',
'CoolWebSearch',
'Gator GAIN',
'"180search Assistant"',
'Transponder *vx2*',
'WannaCry AND (Ransomware OR Attack)',
'"Colonial Pipeline" AND (Ransomware OR Attack)',
'Cryptolocker OR Crypto-locker',
'Dropper Trojan',
'Wiper malware',
'Pharming Attack',
'"Insider Threat"',
'(Drive-by Download) OR (Drive-by Install)',
'Rootkit',
'Adversarial Attack',
'"Data Poisoning"',
'Deepfake',
'Deeplocker OR (Deep-locker) OR "Deep locker"',
'"Supply Chain" AND Attack',
'IoT Device Attack',
'(KEYLOG* Attack) OR (KEYSTROKE LOG Attack)',
'"DNS Tunnel*"',
'"Session Hijacking" OR "cookie hijacking" OR "cookie poisoning"',
'URL Attack',
'"Unknown Attack" AND (Security OR Cyber*)'
]


# this list will be used to count the number of mentions of each attack type by the harvester (attack keywords list)
mention_list=[('Denial-of-Service attack' , 'DDoS'),
              ('Phishing',),
              ('Ransomware',),
              ('Password',),
              ('SQL Injection','SQLI'),
              ('Account Hijack','Account Takeover'),
              ('Deface',),
              ('Trojan',),
              ('Vulnerability'),
              ('Zero-day',),
              ('Advanced Persistent Threat','APT Attack'),
              ('cross-site scripting',),
              ('Malware', 'TROJAN', 'SPYWARE', 'MALVERTIS',  'RANSOMWARE', 'VIRUS', 'WORM', 'KEYLOG', 'KEYSTROKE LOG', 'MALICIOUS CODE', 'MALICIOUS SOFTWARE', 'ADWARE', 'ROOTKIT', 'BOTNET', 'BACKDOOR'),
              ('Data Breach', 'Data Leak', 'information leak', 'Data SPILL', 'unintentional information disclos'),
              ('Disinformation', 'Misinformation', 'false information', 'misleading information', 'Deepfake'),
              ('Targeted Attack',),
              ('Adware',),
              ('Brute Force',),
              ('Malvertis',),
              ('Backdoor',),
              ('Botnet',),
              ('Cryptojack',),
              ('Worm',),
              ('Spyware',),
              ('Man-in-the-middle', 'MITM'), 
              ('DNS Spoof', 'DNS cache poison'),
              ('Pegasus',),
              ('CoolWebSearch',),
              ('Gator Spyware','GAIN Spyware','GAIN) Spyware','Gator (GAIN)','Gator/GAIN','Gator GAIN'),
              ('180search Assistant',),
              ('Transponder vx2','Transponder Spyware','vx2 Spyware','vx2) Spyware','Transponder/vx2', 'Transponder (vx2)'),
              ('WannaCry',),
              ('Colonial Pipeline',),
              ('Cryptolocker', 'Crypto-locker'),
              ('Dropper',),
              ('Wiper',),
              ('Pharming',),
              ('Insider Threat',),
              ('Drive-by',),
              ('Rootkit',),
              ('Adversarial Attack',),
              ('Data Poisoning',),
              ('Deepfake',),
              ('Deeplocker','Deep locker'),
              ('Supply Chain',),
              ('IoT Device','IoT Attack','Attack on IoT','Attack the IoT'),
              ('KEYLOGGER','KEYSTROKE LOG'),
              ('DNS Tunnel',),
              ('Session Hijack','Hijack Session','Hijack the Session', 'cookie hijack', 'cookie poison'),
              ('URL manipul' , 'URL tamper' , 'URL interpret' , 'URL Attack'),
              ('Unknown Attack',)
             ]                       
              



years_list=list(range(2011,2023));
month_list=['January','February','March','April','May','June','July','August','September','October','November','December']

empty='Result set was empty'

search_results_list=[]

#Part 1 - collect the URLs of all relevant documents (for each month and each attack type)
#The search is within the title, abstract, and keywords
for year in years_list:
    print ("searching in year:", str(year))
    
    for month in month_list:

        #Our dataset starts from July 2011
        if year==2011 and month in month_list[:6]:
            continue;

        
        print ("searching in year/month:", str(year),"/",month)
        slsm=[]
        for index,attack in enumerate(attack_list):
            slm=[]
            print("searching for ",attack)
            doc_srch = ElsSearch("TITLE("+attack+") OR ABS("+attack+") OR KEY("+attack+") AND PUBDATETXT("+month+" "+str(year)+")",'scopus')
            doc_srch.execute(client, get_all = True)
            if(empty in str(doc_srch.results)):
                #print(doc_srch.results)
                slsm.append(slm);
                continue
            else:
                #print("got" , len(doc_srch.results))
                for col in doc_srch.results:
                    slm.append(col)
                slsm.append(slm);
        search_results_list.append(slsm)
        
  

if 'search_results_list' in locals():
    print("search_results_list is in memory.")
    print(f"Length of search_results_list: {len(search_results_list)}")
    print(f"First element: {search_results_list[0]}")
else:
    print("search_results_list is not in memory.")


# Increase timeout and add a check to skip URLs where the element is missing
print("--------------------------Harvesting Time ---------------------------")
data = []
for index, row in enumerate(search_results_list):
    print(f"Out of {len(search_results_list)}, Harvesting for column {index}")
    attack_map = {mention[0]: 0 for mention in mention_list}

    for j, col in enumerate(row):
        for k, result in enumerate(col):
            url = result['link'][2]['@href']
            print(f"Processing URL: {url}")

            try:
                driver.get(url)
                print("Successfully loaded the URL.")
            except Exception as e:
                print(f"Failed to load URL {url}: {e}")
                continue

            try:
                element_present = EC.presence_of_element_located((By.CLASS_NAME, 'Highlight-module__akO5D'))
                WebDriverWait(driver, 40).until(element_present)
            except TimeoutException:
                print(f"Element not found, skipping this URL: {url}")
                continue

            try:
                keyword_elements = driver.find_elements(By.TAG_NAME, "button")
                for element in keyword_elements:
                    if 'Indexed keywords' in element.text:
                        element.click()
                        break

                text_elements = driver.find_elements(By.CLASS_NAME, "Highlight-module__akO5D")

                for mention in mention_list:
                    totalQ = sum(element.text.replace("-", " ").upper().count(x.upper().replace("-", " ")) for x in mention for element in text_elements)
                    attack_map[mention[0]] += totalQ
                    print(f"Count for {mention}: {attack_map[mention[0]]}")

            except Exception as e:
                print(f"Error processing elements on page for URL {url}: {e}")
                continue

    # Save after each URL is processed to avoid losing data
    with open('Attacks_NoM_January_2012.txt', 'a') as file_object:
        for key in attack_map:
            file_object.write(f"{key}: {attack_map[key]}\n")
    data.append(attack_map)
