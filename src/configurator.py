import sys
from ast import literal_eval


for arg in sys.argv[1:]:
    if not arg.startswith("--") or "=" not in arg:
        raise ValueError(f"Expected --key=value, got: {arg}")

    key, value = arg[2:].split("=", 1)
    if key not in globals():
        raise ValueError(f"Unknown config key: {key}")
    try:
        value = literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    if type(value) is not type(globals()[key]):
        raise TypeError(f"{key} expects {type(globals()[key]).__name__}, got {type(value).__name__}")

    print(f"Overriding: {key} = {value}")
    globals()[key] = value
