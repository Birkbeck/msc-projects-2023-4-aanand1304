
import sys
import csv
from itertools import zip_longest


def is_leap_year(year):
    """Determine whether a year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

                    

months=['01','02','03','04','05','06','07','08','09','10','11','12']
c=dict()


countries=['US','GB','CA','AU','UA','RU','FR','DE','BR','CN','JP','PK',
           'KP','KR','IN','TW','NL','ES','SE','MX','IR','IL','SA','SY',
           'FI','IE','AT','NO','CH','IT','MY','EG','TR','PT','PS','AE','?','ALL']




attacks=[
'DDoS',
'Phishing',
'Ransomware',
'Password Attack',
'SQL Injection',
'Account Hijacking',
'Defacement', 
'Trojan', 
'Vulnerability',
'Zero-day',
'Advanced persistent threat',
'XSS',
'Malware', 
'Data Breach', 
'Disinformation/Misinformation',
'Targeted Attack',
'Adware',
'Brute Force Attack',
'Malvertising',
'Backdoor',
'Botnet',
'Cryptojacking',
'Worms',
'Spyware',
'Unknown',
'Others'
]


periods=[]
day_counter=0;
days_key='';
for attack in attacks:
    for country in countries:
        key=attack+'-'+country
        c[key]=dict()
        for year in range(2023,2025):
            for month in months:
                for day in range(1,32):
                    if year == 2024 and int(month) > 4:  # Skip months after April 2024
                        continue;
                    if month=='02' and day>28 and not is_leap_year(year):
                        continue;
                    if month=='02' and day>29:
                        continue;
                    if day>30 and (month=='04' or month=='06' or month=='09' or month=='11'):
                       continue;
                       
                    a=''
                    if day<10:
                        a='0'
                    date=a+str(day)+"/"+month+"/"+str(year);
                    c[key][date]=0;
                    if(date not in periods):
                        periods.append(date)
                    print(key+'-'+date);


data2 = []

# Open the CSV file with error handling for encoding issues
with open('Hackmageddon.csv', 'r', encoding='utf-8', errors='replace') as datafile:
    reader = csv.reader(datafile)
    for line_number, row in enumerate(reader, start=1):
        # Try processing the line and handle potential errors
        try:
            # Strip whitespace from the 4th column (index 3)
            row[3] = row[3].strip()
            data2.append(row)
        except UnicodeDecodeError as e:
            print(f"Error decoding line {line_number}: {row}")
            print(e)
            break
        except IndexError as e:
            print(f"Error processing line {line_number}: {row} - {e}")
            break

# Now we have cleaned and stripped data in data2
data_cn = []
for line in data2:
    # Split the 4th column by '-' and gather unique country names
    data_l = line[3].split('-')
    for cn in data_l:
        if cn not in data_cn:
            data_cn.append(cn)

index = 0
cleaned_data = []
for line in data2:
    index += 1
    # Validate the row format
    if len(line) != 4 or (len(line[3]) > 2 and '-' not in line[3]):
        print(f"Problematic row at index {index}: {line}")
        print("oops")
        sys.exit()

    if line[0] == '' or (line[1] == '' and line[2] == '') or line[3] == '':
        print(f"Empty or malformed row at index {index}: {line}")
        print("removing empty rows...")
        sys.exit()
    else:
        cleaned_data.append(line)

# Update the original data with cleaned data
data = cleaned_data

print('Data has', len(data), 'rows')

##################### Counting #####################

malwares=['MALWARE', 'WIPER','TROJAN','DROPPER','SPYWARE','PEGASUS','MALVERTIS', 'RANSOMWARE', 'VIRUS','WORM','KEYLOG', 'KEYSTROKE LOG', 'MALICIOUS CODE','MALICIOUS SOFTWARE','ADWARE','ROOTKIT', 'BOT', 'BACKDOOR']

unk=0
for d in data:
    h_date=d[0].strip()
    #not included
    text=d[1]+d[2]
    attack=d[2]
    country_list_comma=d[3]
    description=d[1]
    text=text.upper().replace("-"," ")
    attack=attack.upper().replace("-"," ")
    description=description.upper().replace("-","")
    known=False
    unk_increased=False
    
    if('>1' in country_list_comma):
        country_list=countries[:-1];#execluding all
    else:
        country_list=country_list_comma.split('-');
    
    for country in country_list:
        
        if country not in countries:
            continue
    
        print('Filling table...date:'+h_date);
        known=False
        
        
        if('DDOS' in text or 'DENIAL OF SERVICE' in text):
            c['DDoS-'+country][h_date]+=1
            c['DDoS-ALL'][h_date]+=1
            known=True
                
        if('PHISHING' in text):
            c['Phishing-'+country][h_date]+=1
            c['Phishing-ALL'][h_date]+=1
            known=True
                
        if('RANSOMWARE' in text):
            c['Ransomware-'+country][h_date]+=1
            c['Ransomware-ALL'][h_date]+=1
            known=True
        
        if('PASSWORD' in text):
            c['Password Attack-'+country][h_date]+=1
            c['Password Attack-ALL'][h_date]+=1
            known=True
            
        if('SQLI' in text or 'SQL I' in text):
            c['SQL Injection-'+country][h_date]+=1
            c['SQL Injection-ALL'][h_date]+=1
            known=True
        
        if('ACCOUNT HIJACK' in text or 'ACCOUNT TAKE' in text):
            c['Account Hijacking-'+country][h_date]+=1
            c['Account Hijacking-ALL'][h_date]+=1
            known=True
        
        if('DEFACE' in text):
            c['Defacement-'+country][h_date]+=1
            c['Defacement-ALL'][h_date]+=1
            known=True

        if('TROJAN' in text or 'DROPPER' in text):
            c['Trojan-'+country][h_date]+=1
            c['Trojan-ALL'][h_date]+=1
            known=True

        if('VULNERABILITY' in text or '0 DAY' in attack or 'ZERO DAY' in text):
            c['Vulnerability-'+country][h_date]+=1
            c['Vulnerability-ALL'][h_date]+=1
            known=True

        if('0 DAY' in attack or 'ZERO DAY' in text):
            c['Zero-day-'+country][h_date]+=1
            c['Zero-day-ALL'][h_date]+=1
            known=True
        
        if('APT' in attack or 'ADVANCED PERSISTENT THREAT' in text):
            c['Advanced persistent threat-'+country][h_date]+=1
            c['Advanced persistent threat-ALL'][h_date]+=1
            known=True            

        if('XSS' in text or 'CROSS SITE SCRIPT' in text):
            c['XSS-'+country][h_date]+=1    
            c['XSS-ALL'][h_date]+=1 
            known=True

        for malware_keyword in malwares:
            if (malware_keyword in text):
                c['Malware-'+country][h_date]+=1
                c['Malware-ALL'][h_date]+=1
                known=True
                break;
                
        if 'BREACH' in text or 'LEAK' in text or 'SPILL' in text or 'EXPOSE' in text:
            c['Data Breach-'+country][h_date]+=1
            c['Data Breach-ALL'][h_date]+=1
            known=True

        if 'DISINFORMATION' in text or 'MISINFORMATION' in text or 'FALSE INFORMATION' in text or 'MISLEADING' in text:
            c['Disinformation/Misinformation-'+country][h_date]+=1
            c['Disinformation/Misinformation-ALL'][h_date]+=1 
            known=True
    
        if('TARGETED ATTACK' in attack):
            c['Targeted Attack-'+country][h_date]+=1
            c['Targeted Attack-ALL'][h_date]+=1
            known=True
            
        if ('ADWARE' in text):
            c['Adware-'+country][h_date]+=1
            c['Adware-ALL'][h_date]+=1
            known=True
            
        if ('BRUTE FORCE' in text):
            c['Brute Force Attack-'+country][h_date]+=1
            c['Brute Force Attack-ALL'][h_date]+=1
            known=True
        
        if ('MALVERTIS' in text):
            c['Malvertising-'+country][h_date]+=1
            c['Malvertising-ALL'][h_date]+=1
            known=True
        
        if ('BACKDOOR' in text):
            c['Backdoor-'+country][h_date]+=1
            c['Backdoor-ALL'][h_date]+=1
            known=True
        
        if ('BOTNET' in text):
            c['Botnet-'+country][h_date]+=1
            c['Botnet-ALL'][h_date]+=1
            known=True

        if ('CRYPTOJACK' in text or 'CRYPTO JACK' in text):
            c['Cryptojacking-'+country][h_date]+=1
            c['Cryptojacking-ALL'][h_date]+=1
            known=True
        
        if ('WORM' in text):
            c['Worms-'+country][h_date]+=1
            c['Worms-ALL'][h_date]+=1
            known=True 
            
        if ('SPYWARE' in text):
            c['Spyware-'+country][h_date]+=1
            c['Spyware-ALL'][h_date]+=1
            known=True          
                
        if('UNKNOWN' in attack):
            c['Unknown-'+country][h_date]+=1 
            c['Unknown-ALL'][h_date]+=1
            known=True
    
        if(not known):
            c['Others-'+country][h_date]+=1 
            c['Others-ALL'][h_date]+=1 
            if not unk_increased:
                unk+=1
                unk_increased=True
            


fields = ['Attack-Country']
for period in periods:
    fields.append(period)

with open('NoI_daily.csv', 'w',newline='') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, fields )
    w.writeheader()
    for key,val in c.items():
        row = {'Attack-Country': key}
        row.update(val)
        w.writerow(row)

# Read the CSV file and transpose the data
original_data = []
with open('NoI_daily.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        original_data.append(row)

transposed_data = list(zip_longest(*original_data, fillvalue=''))  # Transpose the data

# Write the transposed data back to the original CSV file
with open('NoI_daily.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in transposed_data:
        writer.writerow(row)
         
           