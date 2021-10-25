import configparser
import os

def get_config():
    parser = configparser.ConfigParser()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config.txt')
    parser.read(path)
    return parser['DEFAULT']

def set_config(configs):
    assert(type(configs) == dict)
    parser = configparser.ConfigParser()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config.txt')
    parser.read(path)
    for key in configs.keys():
        if key in parser['DEFAULT']:
            parser.set('DEFAULT', key, str(configs[key]))

    parser.write(open(path, 'w'))
