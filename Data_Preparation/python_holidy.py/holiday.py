from datetime import date
import holidays
import csv

def is_leap_year(year):
    """Determine whether a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

countries = ['US', 'GB', 'CA', 'AU', 'UA', 'RU', 'FR', 'DE', 'BR', 'CN', 'JP', 'PK',
             'KP', 'KR', 'IN', 'TW', 'NL', 'ES', 'SE', 'MX', 'IR', 'IL', 'SA', 'SY',
             'FI', 'IE', 'AT', 'NO', 'CH', 'IT', 'MY', 'EG', 'TR', 'PT', 'PS', 'AE']

H = []

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# Create header row with date columns from July 2011 to April 2024
h = ['date']
for year in range(2011, 2024):
    for month in months:
        # Skip months before July 2011
        if int(month) < 7 and year == 2011:
            continue
        # Stop after April 2024
        if int(month) > 4 and year == 2024:
            break
        date_s = month + "/" + str(year)
        h.append(date_s)

H.append(h)

missing = []

for country in countries:
    h = [country + '_holiday']
    for year in range(2011, 2024):
        for month in months:
            # Skip months before July 2011
            if int(month) < 7 and year == 2011:
                continue
            # Stop after April 2024
            if int(month) > 4 and year == 2024:
                break

            counter = 0

            for day in range(1, 32):
                # Skip invalid days for specific months
                if month == '02' and day > 28 and not is_leap_year(year):
                    continue
                if month == '02' and day > 29:
                    continue
                if day > 30 and month in ['04', '06', '09', '11']:
                    continue

                try:
                    c_holidays = holidays.country_holidays(country)  # this is a dict
                except:
                    if country not in missing:
                        missing.append(country)
                    continue

                if date(year, int(month), day) in c_holidays:
                    counter += 1

            h.append(counter)

    H.append(h)
    print('Added holidays of', country)

# Write the collected holiday data to a CSV file
with open('PH_July2011_April2024.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(list(map(list, zip(*H))))

print(f"Missing countries (if any): {missing}")
