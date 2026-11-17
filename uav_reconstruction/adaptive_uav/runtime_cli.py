"""Lightweight dispatch shared by existing command-line entry points."""

def requested_mode(arguments):
    for index,value in enumerate(arguments):
        if value.startswith('--mode='):return value.split('=',1)[1]
        if value=='--mode' and index+1<len(arguments):return arguments[index+1]
    return 'legacy'
