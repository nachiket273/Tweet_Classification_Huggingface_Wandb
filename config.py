import configparser

def get_config():
    parser = configparser.ConfigParser()
    parser.read('config.txt')
    return parser['DEFAULT']

def set_config(configs):
    assert(type(configs) == dict)
    parser = configparser.ConfigParser()
    parser.read('config.txt')
    for key in configs.keys():
        if key in parser['DEFAULT']:
            parser.set('DEFAULT', key, str(configs[key]))

    parser.write(open('config.txt', 'w'))