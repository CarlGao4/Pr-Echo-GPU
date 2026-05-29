# Copyright (c) 2026 CarlGao4
# Licensed under the MIT License. See LICENSE for details.
# github.com/CarlGao4/Pr-Echo-GPU

import re
import sys
import pathlib

if len(sys.argv) != 2:
    print('Removes "#line ..." directives from the given file, keeping only file name', file=sys.stderr)
    print("Usage: python3 fix_line_path_removal.py <file>", file=sys.stderr)
    sys.exit(1)

filename = sys.argv[1]
with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

def replace_line(match: re.Match) -> str:
    path = match.group(2)
    filename = pathlib.PurePath(path).name
    return f'#line {match.group(1)} "{filename}"'

content = re.sub(r'#line +(\d+) +"(.*?)"', replace_line, content)

with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)
