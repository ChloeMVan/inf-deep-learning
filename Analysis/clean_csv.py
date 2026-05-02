import csv

input_file = '../whole_dataset_splits.csv'
output_file = '../whole_dataset_splits_no.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    for line in infile:
        fixed_line = line.replace('/content/', '/')
        outfile.write(fixed_line)