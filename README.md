# rl-fundamentos

Curso práctico de fundamentos de aprendizaje por refuerzo, en notebooks progresivos, aplicado a
juegos de cartas y a problemas de negocio (despacho de flota). Recorre desde bandits hasta PPO con
pointer attention, self-play, liga de oponentes y currículo generalista → especialista, al estilo del
ganador de la Kaggle PTCG AI Battle Challenge.

Cada lección es un notebook en español con el código completo y funcionando, gráficos, experimentos
"predice antes de correr" (con la respuesta en un bloque plegado) y un criterio de éxito verificado
con intervalos de confianza. Los `.ipynb` están commiteados **ejecutados, con sus salidas**, para poder
leerlos directamente en GitHub. La fuente de cada uno es el `.py` (jupytext, formato percent).

## Índice del curso

| # | Notebook | Tema | Entornos |
|---|----------|------|----------|
| 00 | [`00_interfaz`](notebooks/00_interfaz.ipynb) | La interfaz agente ↔ entorno y el agente aleatorio | Bandit, MiniDespacho, MiniCartas |
| 01 | [`01_bandits`](notebooks/01_bandits.ipynb) | Bandits: explorar vs explotar (ε-greedy, UCB) | Bandit |
| 02 | [`02_medir`](notebooks/02_medir.ipynb) | Medir sin engañarse: IC, tamaño de muestra, evaluación pareada, Elo, sesgo de asiento | los tres |
| 03 | [`03_mdp_bellman`](notebooks/03_mdp_bellman.ipynb) | MDP, ecuación de Bellman y value iteration | MiniDespacho diminuto |
| 04 | [`04_qlearning`](notebooks/04_qlearning.ipynb) | Q-learning tabular: aprender sin el modelo | MiniDespacho diminuto |
| 05 | [`05_dqn`](notebooks/05_dqn.ipynb) | De tabla a red neuronal: DQN | MiniDespacho |
| 06 | [`06_reinforce`](notebooks/06_reinforce.ipynb) | Policy gradient: REINFORCE con baseline | MiniDespacho |
| 07 | [`07_ppo`](notebooks/07_ppo.ipynb) | Actor-crítico, GAE y PPO | MiniDespacho |
| 08 | [`08_masking_pointer`](notebooks/08_masking_pointer.ipynb) | Acciones variables: masking y pointer attention | MiniDespacho, MiniCartas |
| 09 | [`09_transformer`](notebooks/09_transformer.ipynb) | Entidades como tokens y Transformer | MiniDespacho, MiniCartas |
| 10 | [`10_selfplay`](notebooks/10_selfplay.ipynb) | Dos jugadores: self-play e información oculta con hechos y cotas | MiniCartas |
| 11 | [`11_liga_arena`](notebooks/11_liga_arena.ipynb) | Liga de oponentes, arena y matriz de counters | MiniCartas |
| 12 | [`12_curriculo_negocio`](notebooks/12_curriculo_negocio.ipynb) | Currículo generalista → especialista y llevarlo a negocio (sim-to-real, errores del simulador) | MiniDespacho, MiniCartas |

## Estructura

- `rlf/`: el código de apoyo. `core.py` (interfaz común: observación + lista de opciones legales →
  índice), `envs/` (Bandit, MiniDespacho, MiniCartas), `agents/` (tabular, DQN, REINFORCE, PPO,
  baselines), `nets/` (MLP, pointer, Transformer de tokens), `eval/` (estadística, arena).
- `notebooks/`: las 13 lecciones (`.py` + `.ipynb` ejecutado).
- `tests/`: pruebas de cada pieza.
- `openspec/changes/curso-rl/`: el brief (`proposal.md`), el diseño (`design.md`, con el registro de
  calibración en la sección 10) y la cola de tareas (`tasks.md`).

## Cómo correrlo

Requiere Python 3.12.

```bash
pip install -e ".[dev]"
pytest -q                                              # tests
jupytext --to ipynb --set-kernel python3 notebooks/07_ppo.py
pytest --nbmake --overwrite --nbmake-timeout=600 notebooks/07_ppo.ipynb   # ejecuta y guarda las salidas
```

`RLF_TRAIN_SEED` (0 por defecto) cambia la semilla de entrenamiento de un notebook; las semillas de
evaluación son fijas y disjuntas de las de entrenamiento. Cada notebook corre en CPU en 2 a 10 minutos.

## Verificación integral

```bash
pytest -q
for n in notebooks/*.ipynb; do pytest --nbmake --nbmake-timeout=600 "$n" -q || exit 1; done
```
