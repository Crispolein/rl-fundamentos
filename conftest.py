import os

# Tests and notebooks run on CPU unless the caller forces another device.
os.environ.setdefault("RLF_DEVICE", "cpu")
