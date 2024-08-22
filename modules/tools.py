import bleach
import os


with open("formats/default.txt", 'r') as f:
    default_format = f.read()
with open("formats/task_focus.txt", 'r') as f:
    task_focus_format = f.read() 

def clean_text(text):
    if text is None:
        return ""
    return bleach.clean(text, strip=True).replace('&nbsp;', '').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

def formatter(ticket, formatting):
    if formatting == "task_focus":
        return task_focus_format.format(**ticket)
    else:
        return default_format.format(**ticket)
    