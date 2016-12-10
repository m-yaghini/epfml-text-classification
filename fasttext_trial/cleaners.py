# cleaner functions
#

import os


def clean_duplicates(in_file, out_file):
    lines_seen = set()  # holds lines already seen
    with open(out_file, "w") as outfile:
        with open(in_file, "r") as infile:
            for line in infile:
                if line not in lines_seen:  # not a duplicate
                    outfile.write(line)
                    lines_seen.add(line)

def clean_user_tags(in_file, out_file):
    with open(in_file) as infile, open(out_file, 'w') as outfile:
        for line in infile:
            line_out = line.replace('<user>', '')
            line_out = line_out.strip() + '\n'
            outfile.write(line_out)


def clean_data(in_file, out_file):
    clean_duplicates(in_file, 'temp.txt')
    clean_user_tags('temp.txt', out_file)
    os.remove('temp.txt')
