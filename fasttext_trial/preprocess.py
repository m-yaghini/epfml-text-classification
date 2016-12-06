# Pre-processing functions


def add_label(in_file, label, out_file):
    with open(in_file) as infile, open(out_file, 'w') as outfile:
        for line in infile:
            line_out = '__label__' + label + ' ' + line
            outfile.write(line_out)