"""Every (name, factory) pair here is run through the contract tests in test_env_contract.py."""
from rlf.envs.bandit import Bandit
from rlf.envs.despacho import DespachoEnv
from tests.toy_env import ToyDuel, ToyEnv

ENVS = {
    "toy1": lambda: ToyEnv(1),
    "toy2": lambda: ToyEnv(2),
    "toyduel": lambda: ToyDuel(),
    "bandit": lambda: Bandit(k=10, horizon=300),
    "bandit_small": lambda: Bandit(k=3, horizon=20),
    "despacho_diminuto": lambda: DespachoEnv.from_size("diminuto"),
    "despacho_mediano": lambda: DespachoEnv.from_size("mediano"),
    "despacho_grande": lambda: DespachoEnv.from_size("grande"),
}
