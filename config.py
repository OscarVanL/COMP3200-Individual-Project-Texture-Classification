import sys


class GlobalConfig:
    """
    Implementation based on: https://stackoverflow.com/a/43941592/6008271
    """
    __conf = {
        "algorithm": None,
        "dataset": None,
        "train_ratio": 1.0,
        "scale": 1.0,
        "test_scale": None,
        "folds": 100,
        "rotate": False,
        "noise": None,
        "noise_val": None,
        "multiprocess": False,
        "cpu_count": None,
        "examples": False,
        "train_noise": False,
        "ECS": False,
        "debug": False,
        "CWD": None
    }

    __setters = ["algorithm", "dataset", "train_ratio", "scale", "test_scale", "folds", "rotate", "noise", "noise_val", "multiprocess", "cpu_count", "examples", "train_noise", "ECS", "debug", "CWD"]

    @staticmethod
    def get(name):
        try:
            return GlobalConfig.__conf[name]
        except KeyError as e:
            print(e)
            print("No known setting called", name, file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def set(name, value):
        if name in GlobalConfig.__setters:
            GlobalConfig.__conf[name] = value
        else:
            raise NameError("Value name not accepted in set() method")
