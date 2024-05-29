_CONFIGS = {}


def register_config(name=None):
    def _register(func):
        def _fn(*args, **kwargs):
            return func(*args, **kwargs)

        _CONFIGS[name] = _fn
        return _fn

    return _register


def get_config(name):
    return _CONFIGS[name]()
