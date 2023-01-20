
def convert_type(value):
    if value.isdigit():
        return int(value)
    if value.count('.') == 1 and value[-1] == '.':
        return float(value)
    if value.count('.') == 1 and value[0] != '.':
        a, b = value.split('.')
        if a.isdigit() and b.digit():
            return float(value)
    if value in ['false', 'False']:
        return False
    if value in ['true', 'True']:
        return True

    return value

def recursion_set_config(cfg, key, value):
    if len(key) == 1:
        cfg[key[0]] = convert_type(value)
    else:
        recursion_set_config(cfg[key[0]], key[1:], value)