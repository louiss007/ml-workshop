import sys
import os


def read_multi_lines(infile, num):
    buffer_size = num * 128
    with open(infile, 'r') as fi:
        while True:
            data = fi.read(buffer_size)
            num_lines = data.split('\n')[:num]
            fi.seek(len(num_lines[0]), os.SEEK_SET)
            print "======"
            # print fi.tell()
            print "\n".join(num_lines)
            if not fi.read(buffer_size):
                break


infile = sys.argv[1]
read_multi_lines(infile, 3)
