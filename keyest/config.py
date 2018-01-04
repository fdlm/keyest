from os.path import expanduser
import socket

if socket.gethostname() == 'nowhere-man':
    EXPERIMENT_ROOT = expanduser('~/CP/experiments/keyest')
    DATASET_DIR = expanduser('~/CP/data')
else:
    EXPERIMENT_ROOT = expanduser('~/experiments/keyest')
    DATASET_DIR = expanduser('~/data')

CACHE_DIR = expanduser('~/.tmp')
