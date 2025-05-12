"""
this is basically for test csv setup and literally nothing else
"""
import csv

filepath = r"X:/LLM/BatchGrader/tests/input/large.csv"


def replace_csv_values(filepath, placeholder="PLACEHOLDER"):
    rows = []
    with open(filepath, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            new_row = [placeholder] * len(row)
            rows.append(new_row)
    with open(filepath, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

replace_csv_values(filepath)
print(f"All values in '{filepath}' have been replaced with 'PLACEHOLDER'.")