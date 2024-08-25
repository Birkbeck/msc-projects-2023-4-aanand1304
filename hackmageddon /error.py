import csv

file_path = 'Hackmageddon.csv'
line_number = 0

with open(file_path, 'r', encoding='utf-8', errors='replace') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line_number += 1
        try:
            # Simulate processing the line to check for errors
            _ = [str(column) for column in line]
        except UnicodeDecodeError as e:
            print(f"Error decoding line {line_number}: {line}")
            print(e)
