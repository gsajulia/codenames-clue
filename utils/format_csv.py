import pandas as pd
import csv

def convert_floats_to_strings(input_file, output_file):
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        
        rows = []
        for row in reader:
            new_row = [f'="{item}"' if is_float(item) else item for item in row]
            rows.append(new_row)
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

def is_float(value):
    try:
        float(value)
        return '.' in value # To format similarity
    except ValueError:
        return False


convert_floats_to_strings("v6_for_3_game.csv", "v6_for_3_game_formated.csv")