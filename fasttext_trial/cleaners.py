# cleaner functions
#


def clean_duplicates(in_file, out_file):
    lines_seen = set()  # holds lines already seen
    outfile = open(out_file, "w")
    for line in open(in_file, "r"):
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()


def clean_user_tags(in_file, out_file):
    with open(in_file) as infile, open(out_file, 'w') as outfile:
        for line in infile:
            line_out = line.replace('<user>', '')
            line_out = line_out.strip() + '\n'
            outfile.write(line_out)
