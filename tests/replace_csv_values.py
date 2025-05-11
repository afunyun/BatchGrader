import csv

filepath = X:/LLM/BatchGrader/tests/input/large.csv


def replace_csv_values(filepath, placeholder="PLACEHOLDER"):
    rows = []
    # Read the existing data to understand its structure (number of columns per row)
    with open(filepath, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            # Create a new row consisting only of placeholders,
            # matching the original number of columns for that row.
            new_row = [placeholder] * len(row)
            rows.append(new_row)

    # Write the modified data (all placeholders) back to the same file
    with open(filepath, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

csv_file_path = r'x:\LLM\BatchGrader\tests\input\large.csv'
replace_csv_values(csv_file_path)
print(f"All values in '{csv_file_path}' have been replaced with 'PLACEHOLDER'.")