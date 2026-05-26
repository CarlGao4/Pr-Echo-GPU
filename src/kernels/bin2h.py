#!/usr/bin/env python3
"""
bin2h.py - Convert a binary file into a C header with
    unsigned char const kName[] = { ... };
    unsigned int const kName_Size = ...;

Usage:
    python bin2h.py -i input.bin -o output.h -n kMyBlob
"""

import sys, getopt

def parse(argv):
    inputfile = ''
    outputfile = ''
    name = 'kBlob'
    append = False
    try:
        opts, _ = getopt.getopt(argv[1:], "hi:o:n:a", ["ifile=", "ofile=", "name=", "append"])
    except getopt.GetoptError:
        printhelp(argv[0])
    for opt, arg in opts:
        if opt == '-h':
            printhelp(argv[0])
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-a", "--append"):
            append = True
    return inputfile, outputfile, name, append

def printhelp(command):
    print(f"{command} -i <inputFile> -o <outputFile> -n <name> [-a]")
    sys.exit(2)

def bin2h(path, name, out):
    with open(path, 'rb') as f:
        data = f.read()
    out.write(f"namespace {{ unsigned char const {name}[] = {{\n")
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        out.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
    out.write(f"}}; unsigned int const {name}_Size = {len(data)}u;\n}}\n")

def main():
    inputfile, outputfile, name, append = parse(sys.argv)
    if not inputfile:
        printhelp(sys.argv[0])
    mode = 'a' if append else 'w'
    with open(outputfile, mode) if outputfile else sys.stdout as out:
        bin2h(inputfile, name, out)

if __name__ == '__main__':
    main()
