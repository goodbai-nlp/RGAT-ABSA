"""
Helper functions.
"""
import os
import json
import six
import argparse
import subprocess

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)

def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def unpack_raw_data(raw_data, batch_size=32):
    unpacked = []
    for d in raw_data: 
        for a in d['aspects']:
            unpacked.append({'token':d['token'], 'aspect':a, 'polarity':a['polarity']})
    batches = [unpacked[i:i+batch_size] for i in range(0, len(unpacked), batch_size)]
    unpacked = []
    for batch in batches:
        lens = [len(x['token']) for x in batch]
        temp = [t[0] for t in list(sorted(zip(batch,lens,list(range(len(lens)))), key=lambda x:(x[1],x[2]), reverse=True))]
        unpacked.extend(temp)
    return unpacked

