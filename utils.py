import re
from subprocess import call


def speak(text):
    cmd_beg = 'say "'
    cmd_end = '"'
    call([cmd_beg + text + cmd_end], shell=True)


def read_label_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
            ret[int(pair[0])] = pair[1].strip()
        else:
            ret[row_number] = content.strip()
    return ret
