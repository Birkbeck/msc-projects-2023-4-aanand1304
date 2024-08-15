import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
import json
import csv
from datetime import datetime, timedelta
import time
import random

# Load configuration
con_file = open("config.json")  # Place your Elsevier key in this file
config = json.load(con_file)
con_file.close()

# Initialise Elsevier client
client = ElsClient(config['apikey'])

# Configure Brave options
options = Options()
options.binary_location = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"  # Adjust path to your Brave installation
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
# Uncomment the line below if you want to run in headless mode (no GUI)
# options.add_argument('--headless')

# Create a new instance of the Brave driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def process_attack_for_month(year, month, attack):
    print(f"Searching for {attack} in {month} {year}")
    doc_srch = ElsSearch(f"TITLE({attack}) OR ABS({attack}) OR KEY({attack}) AND PUBDATETXT({month} {year})", 'scopus')
    doc_srch.execute(client, get_all=True)
    if empty in str(doc_srch.results):
        return []
    else:
        return doc_srch.results



def harvest_data_for_url(result, attack_map, count_tracker):
    url = result['link'][2]['@href']
    print(f"Processing URL: {url}")

    retries = 3  # Number of retries
    for attempt in range(retries):
        try:
            driver.get(url)
            print("Successfully loaded the URL.")
            
            element_present = EC.presence_of_element_located((By.TAG_NAME, 'h1'))
            WebDriverWait(driver, 100).until(element_present)

            title_element = driver.find_element(By.TAG_NAME, 'h1')
            title = title_element.text
            print(f"Title found: {title}")

            for mention in mention_list:
                text_elements = driver.find_elements(By.CLASS_NAME, "Highlight-module__akO5D")
                totalQ = sum(element.text.replace("-", " ").upper().count(x.upper().replace("-", " ")) for x in mention for element in text_elements)
                attack_map[mention[0]] += totalQ
                print(f"Count for {mention}: {attack_map[mention[0]]}")

            count_tracker['completed'] += 1
            print(f"Completed harvesting for {count_tracker['completed']} out of {count_tracker['total']} URLs.")
            break  # Exit the retry loop if successful

        except TimeoutException:
            print(f"Element not found, retrying... ({attempt + 1}/{retries})")
            time.sleep((2 ** attempt) + random.random())  # Exponential backoff
            if attempt == retries - 1:
                print(f"Max retries reached for URL: {url}. Skipping this URL.")
        except Exception as e:
            print(f"Failed to load URL {url} on attempt {attempt + 1}/{retries}: {e}")
            time.sleep((2 ** attempt) + random.random())  # Exponential backoff
            if attempt == retries - 1:
                print(f"Max retries reached for URL: {url}. Skipping this URL.")

        # Add a random delay between each URL processing to avoid being detected as a bot
        time.sleep(random.uniform(2, 5))  # Delay between 2 to 5 seconds

def process_month(year, month):
    attack_map = {mention[0]: 0 for mention in mention_list}
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_attack_for_month, year, month, attack) for attack in attack_list]
        for future in concurrent.futures.as_completed(futures):
            slm = future.result()
            all_results.extend(slm)

            # Adding a short delay between processing each attack to avoid rate limiting
            time.sleep(random.uniform(1, 3))  # Delay between 1 to 3 seconds

    count_tracker = {'completed': 0, 'total': len(all_results)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(harvest_data_for_url, result, attack_map, count_tracker) for result in all_results]
        for _ in concurrent.futures.as_completed(futures):
            pass

    return attack_map


# Generate list of months from January 2023 to April 2024
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 4, 30)
current_date = start_date

years_and_months = []
while current_date <= end_date:
    years_and_months.append((current_date.year, current_date.strftime('%B')))
    current_date += timedelta(days=32)
    current_date = current_date.replace(day=1)

# This list will be used in the query of Elsevier (attack list)
attack_list = [
    ('Vulnerability',)
]

# This list will be used to count the number of mentions of each attack type by the harvester (attack keywords list)
mention_list = [
    ('Vulnerability',)
]

empty = 'Result set was empty'

csv_header = ['Month/Year'] + [mention[0] for mention in mention_list]
csv_rows = []

for year, month in years_and_months:
    print(f"Processing {month} {year}")
    attack_map = process_month(year, month)

    row = [f"{month}-{year}"] + [attack_map[mention[0]] for mention in mention_list]
    csv_rows.append(row)

# Write all data to a single CSV file
with open('Attacks_NoM_Jan2023_Apr2024.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csv_header)
    csvwriter.writerows(csv_rows)

print("All data has been processed and written to the CSV file.")
