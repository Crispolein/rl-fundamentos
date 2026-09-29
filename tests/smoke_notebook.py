# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# %%
import numpy as np
import rlf
from rlf.core import seed_everything

seed_everything(0)
assert np.random.rand() < 1.0
print("smoke ok", rlf.__name__)
