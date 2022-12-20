

def recursion_set_config(cfg, key, value):
    if len(key) == 1:
        cfg[key[0]] = value
    else:
        recursion_set_config(cfg[key[0]], key[1:], value)