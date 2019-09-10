# config类转dict
def object2dict(obj):
    dict = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            dict[name] = value
    return dict
