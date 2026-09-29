"""Every (name, factory) pair here is run through the contract tests in test_env_contract.py."""
from tests.toy_env import ToyEnv

ENVS = {
    "toy1": lambda: ToyEnv(1),
    "toy2": lambda: ToyEnv(2),
}
