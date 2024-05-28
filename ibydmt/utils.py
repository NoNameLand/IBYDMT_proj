def _register_cls(name, dict=None):
    def _register(cls):
        if name in dict:
            raise ValueError(f"{name} is already registered")

        dict[name] = cls

    return _register


def _get_cls(name, dict=None):
    return dict[name]
