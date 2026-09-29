# Diseño del curso de fundamentos de RL

Este documento desarrolla `proposal.md` (fuente de verdad). Quien implemente no vio la conversación de
diseño: todo lo necesario para escribir código está aquí o en `proposal.md`. Si algo de este documento
contradice `proposal.md`, gana `proposal.md` y hay que avisar a Christian.

Convenciones de lectura: "n" siempre es número de episodios, partidas o problemas **independientes**;
"IC" es el intervalo de confianza del 95 %; "LB" es su límite inferior.

## 0. Versiones elegidas y por qué

Consultadas el 2026-09-29 en la API JSON de PyPI (`https://pypi.org/pypi/<paquete>/json`). Las páginas
de documentación de PyTorch (`docs.pytorch.org`) y de nbmake no fueron accesibles desde el sandbox de
diseño (el proxy las bloquea, HTTP 403), así que **no se leyó documentación de PyTorch**; en su lugar se
verificó comportamiento ejecutando las versiones (abajo). Quien implemente debe consultar la
documentación de la versión instalada antes de usar cualquier API de PyTorch que no aparezca aquí.

| Paquete | Versión fijada | Fecha PyPI | Por qué |
|---|---|---|---|
| Python | 3.12 | — | Lo fija `proposal.md`. `numpy 2.5.3` exige `>=3.12`. `python3.12` existe en el sandbox cloud. `requires-python = ">=3.12,<3.13"`. |
| torch | 2.14.0 | 2026-09-02 | Última estable. Publica ruedas `cp312` para `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64` y `macosx_14_0_arm64`: cubre el CPU Linux de las sesiones cloud y el Apple M5. En Linux la rueda de PyPI es la variante CUDA (`2.14.0+cu130`, descarga grande); funciona en CPU. El índice `download.pytorch.org/whl/cpu` no es alcanzable desde el sandbox, así que **se instala desde PyPI**. |
| numpy | 2.5.3 | 2026-09-06 | Última estable; probada junto a torch 2.14.0 (ver "Verificado"). |
| matplotlib | 3.11.2 | 2026-09-11 | Última estable; requiere `>=3.11`. |
| jupytext | 1.19.5 | 2026-07-21 | Última estable. Soporta formato `percent`. Depende de `nbformat`, `markdown-it-py`, `pyyaml`, `packaging`, `mdit-py-plugins`. |
| nbmake | 1.5.5 | 2024-12-23 | Última versión publicada (no se actualiza desde 2024). Depende de `pytest>=6.1`, `nbclient>=0.6.6`, `ipykernel>=5.4`. Funciona con pytest 9.1.1 (ver "Verificado"). |
| pytest | 9.1.1 | 2026-06-19 | Última estable. |
| ipykernel | 7.3.0 | 2026-06-10 | Necesario para ejecutar notebooks; instala el kernel `python3` dentro del venv. |
| nbclient | (lo resuelve pip; 0.11.0 al diseñar) | 2026-06-05 | Dependencia de nbmake, no se fija a mano. |

Verificado ejecutando en un venv Python 3.12 limpio (no es opinión, es lo que pasó):

1. `import torch, numpy`: `torch 2.14.0+cu130`, `numpy 2.5.3`; `torch.from_numpy(...)` y `.numpy()` funcionan;
   `torch.cuda.is_available()` y `torch.backends.mps.is_available()` devuelven `False` en CPU Linux.
2. `torch.nn.MultiheadAttention(batch_first=True)` con `key_padding_mask` booleano (**`True` = ignorar esa clave**)
   corre sin NaN. `masked_fill(~mask, -inf)` + `softmax` da probabilidad exactamente 0 a lo enmascarado.
3. `jupytext --to ipynb --execute X.py` **falla** con `No kernel found that matches the current python
   executable` aunque `ipykernel` esté instalado. Flujo que sí funciona y que usa todo el curso:
   `jupytext --to ipynb --set-kernel python3 X.py` (genera el `.ipynb` sin salidas) y luego
   `pytest --nbmake --overwrite --nbmake-timeout=600 X.ipynb` (ejecuta y **guarda las salidas** en el
   mismo `.ipynb`). `--nbmake-timeout` y `--overwrite` existen en nbmake 1.5.5.
4. `torch.get_num_threads()` por defecto es 4 en el sandbox; `core.seed_everything` fija los hilos (sección 2).

Fijación: `pyproject.toml` usa `==` para todas las versiones de la tabla (excepto nbclient). Los umbrales
de aprendizaje de las lecciones se calibran contra estas versiones exactas; subirlas es una decisión de
Christian, no de quien implemente.

## 1. Alcance de este cambio y estructura

Igual que `proposal.md`, con estas altas (todas pequeñas y necesarias; marcadas en la sección 9 para que
Christian las apruebe):

```
rlf/
  core.py                 # sección 3
  envs/ bandit.py despacho.py cartas.py
  agents/ baselines.py    # + RandomPolicy, GreedyDespacho, HeuristicCartas
          tabular.py dqn.py reinforce.py ppo.py
  nets/   mlp.py pointer.py tokens.py
  eval/   stats.py arena.py
conftest.py                 # raíz: fija RLF_DEVICE=cpu por defecto
tests/
  test_core.py test_env_contract.py smoke_notebook.py
  envs/ test_bandit.py test_despacho.py test_cartas.py
  eval/ test_stats.py test_arena.py
  agents/ test_baselines.py test_tabular.py test_dqn.py test_reinforce.py test_ppo.py
  nets/ test_mlp.py test_pointer.py test_tokens.py
notebooks/  NN_nombre.py + NN_nombre.ipynb
build/      # ignorado por git; salidas temporales de verificación
```

Nombres de notebooks: `00_interfaz`, `01_bandits`, `02_medir`, `03_mdp_bellman`, `04_qlearning`, `05_dqn`,
`06_reinforce`, `07_ppo`, `08_masking_pointer`, `09_transformer`, `10_selfplay`, `11_liga_arena`,
`12_curriculo_negocio`.

Comandos: todos asumen el venv activado (`. .venv/bin/activate`), que se crea en la primera tarea de `tasks.md` (andamiaje).

## 2. Convenciones transversales

**Semillas.** Todo el azar de un entorno sale de `np.random.default_rng(seed)` creado en `reset(seed)`.
Rangos de semillas: entrenamiento `[0, 10_000_000)`; evaluación `EVAL_SEED_BASE + i` con
`EVAL_SEED_BASE = 10_000_000`. Nunca se evalúa con semillas de entrenamiento. La semilla de entrenamiento de
un notebook es `TRAIN_SEED = int(os.environ.get("RLF_TRAIN_SEED", "0"))`.

**Evaluación pareada.** Al comparar agentes A y B, ambos juegan las mismas semillas de entorno
`EVAL_SEED_BASE + i`, y se reporta el IC de la diferencia por episodio (`paired_diff_ci`), no dos IC
separados. En juegos de dos jugadores cada semilla se juega dos veces, intercambiando asientos (n/2 y n/2).

**Criterio de aprendizaje.** Cada lección declara afirmaciones con umbral. Hay tres tipos y cada afirmación
dice cuál es. (1) *Afirmación de aprendizaje o de comparación estadística*: "X ≥ t" pasa si el **LB del IC del 95 % de X
es ≥ t**, con n ≥ 1000; "X > 0" pasa si LB > 0; "X < 0" pasa si el límite superior del IC es < 0. (2) *Chequeo de
consistencia estadístico* (p. ej. "la media aleatoria es compatible con 0"): pasa si `|media − valor esperado| ≤ 3 errores
estándar` (con 3 y no 1,96 para que una semilla fija no falle el 5 % de las veces por azar). (3) *Chequeo determinista*
(residuos, invarianzas, conteos, cotas): se cumple exactamente, sin IC. El notebook
termina con una celda `## Criterio de la lección` que calcula cada afirmación, la imprime en una tabla y
hace `assert` de todas. `passes_lower` (sección 5.1) lanza `InsufficientSample` si n < 1000: el propio código
impide afirmar con menos.

**Puntaje normalizado** (para comparar aprendices contra dos anclas en MiniDespacho). Con `r_i`, `g_i`, `a_i` los
retornos por semilla `i` del agente aleatorio, de `GreedyDespacho` y del agente, y α ∈ [0, 1],
`d_i = a_i − (1−α)·r_i − α·g_i`; la afirmación "puntaje ≥ α" pasa si el LB del IC de `mean(d_i)` es ≥ 0
(α = 0,5: a mitad de camino entre aleatorio y greedy; α = 1: iguala a greedy). `normalized_score_ci` (5.1).

**Robustez a la semilla de entrenamiento.** El criterio de cada notebook debe pasar con
`RLF_TRAIN_SEED=0` y con `RLF_TRAIN_SEED=1`. La verificación de cada notebook corre ambas (la de semilla 1
sin `--overwrite`; la de semilla 0 con `--overwrite`, que es la que queda commiteada).

**Presupuesto de tiempo.** Límite duro: 10 min en CPU (`--nbmake-timeout=600`). Objetivo de diseño: **≤ 8 min**
por notebook, para dejar 20 % de margen por diferencias de CPU. Cada lección lista su reparto (entrenar + evaluar).
Los presupuestos de entrenamiento se fijan en **pasos de entorno**, no en segundos, para que el resultado sea
reproducible; los minutos de la tabla son la estimación con la que se eligió ese número de pasos. Si al medir un
notebook pasa de 8 min, se reducen pasos y se re-verifica el criterio; si el criterio no pasa con menos
pasos, la tarea se bloquea (regla de `proposal.md`), no se baja el umbral.

**Dispositivo y hilos.** `get_device()` devuelve cuda → mps → cpu; la variable `RLF_DEVICE` lo fuerza
(`pytest` y `nbmake` corren con `RLF_DEVICE=cpu`; `conftest.py` en la raíz del repo lo fija, para que también valga al correr `pytest --nbmake notebooks/...`). Redes pequeñas corren igual
o más rápido en CPU que en MPS; los umbrales se calibran en CPU. `seed_everything` fija `random`, `numpy`,
`torch.manual_seed` y `torch.set_num_threads(int(os.environ.get("RLF_THREADS", min(4, os.cpu_count()))))`.

**Experimentos "predice antes de correr".** Cada experimento tiene: celda markdown con la pregunta, celda de
código que lo ejecuta con n ≥ 1000 y termina en un `assert` sobre lo que afirma la respuesta, y una celda
markdown con `<details><summary>Respuesta esperada</summary>…</details>`. Así una respuesta plegada falsa
rompe `nbmake`.

**Idioma.** Identificadores, docstrings y comentarios de código en inglés; prosa de notebooks y docs en español.

## 3. `rlf/core.py`: la interfaz

Copiada del motor del PTCG: el entorno entrega observación + **lista de opciones legales**; el agente
devuelve el **índice** de una opción. "Índices" (plural en `proposal.md`) es un índice por entorno: en un
entorno simple es un `int`, en la versión en lotes es un arreglo `int64[B]`, uno por entorno del lote.

```python
EVAL_SEED_BASE = 10_000_000

@dataclass(frozen=True)
class EnvSpec:
    name: str
    n_players: int      # 1 (Bandit, MiniDespacho) o 2 (MiniCartas)
    state_dim: int      # dimensión del vector global
    option_dim: int     # dimensión de las features de cada opción
    entity_dim: int     # dimensión de las features de cada entidad (0 si no hay)
    max_options: int    # tope de opciones simultáneas (para rellenar en lotes)
    max_entities: int   # nº fijo de filas de entidades (0 si no hay)
    n_actions: int      # tamaño del espacio global de ids de acción (para tablas y máscaras)
    max_steps: int      # cota superior de decisiones por episodio (test: se respeta)

@dataclass(frozen=True)
class Obs:                       # lo que ve el jugador que debe decidir
    state: np.ndarray            # float32 [state_dim]
    options: np.ndarray          # float32 [n_opts, option_dim]; una fila por opción legal
    option_ids: np.ndarray       # int64   [n_opts]; id global de cada opción, estrictamente creciente
    option_refs: np.ndarray      # int64   [n_opts, 2]; índices de entidad a los que apunta la opción, o -1
    entities: np.ndarray         # float32 [max_entities, entity_dim]; filas vacías en cero
    entity_mask: np.ndarray      # bool    [max_entities]; True = fila con contenido
    player: int                  # jugador que decide ahora (0 si n_players == 1)
    @property
    def n_options(self) -> int   # len(options)

@dataclass(frozen=True)
class Step:
    obs: Obs                     # observación siguiente; si done, tiene n_options == 0
    rewards: np.ndarray          # float64 [n_players]; recompensa acumulada por jugador en esta transición
    done: bool
    info: dict                   # datos de diagnóstico; nunca los usa el agente para decidir

class Env(Protocol):
    spec: EnvSpec
    def reset(self, seed: int | None = None) -> Obs: ...
    def step(self, choice: int) -> Step: ...      # choice ∈ [0, obs.n_options)
    def state_key(self) -> tuple: ...             # estado discreto hashable (solo para tablas; entornos chicos)
```

Contrato de `reset`/`step` (cada punto es un test de `tests/test_env_contract.py`, parametrizado sobre todos
los entornos y tamaños):

- **C1 Determinismo:** mismo `seed` y mismas elecciones ⇒ misma trayectoria (obs, recompensas, done) bit a bit.
  `reset(None)` toma una semilla de `np.random.SeedSequence()` y la guarda en `env.seed`.
- **C2 Legalidad:** mientras no `done`, `obs.n_options ≥ 1` (siempre hay una opción: esperar, terminar turno…).
- **C3 Ids:** `option_ids` estrictamente creciente y dentro de `[0, spec.n_actions)`. `options[j]` corresponde a
  `option_ids[j]`; el agente elige `j`, no el id.
- **C4 Formas y tipos:** `state`, `options`, `entities` son `float32` finitos con las formas del `spec`;
  `entities.shape == (max_entities, entity_dim)` siempre; `n_options ≤ max_options`.
- **C5 Errores:** `step(j)` con `j` fuera de `[0, n_options)` lanza `IndexError`; `step` tras `done` lanza `RuntimeError`.
- **C6 Fin:** `done` ⇒ `obs.n_options == 0`; con política aleatoria, 1000 episodios terminan en ≤ `spec.max_steps` pasos.
- **C7 Multijugador:** `rewards.shape == (n_players,)`; `obs.player` es quien decide; en 1 jugador es siempre 0.
- **C8 Lotes:** `VecEnv` con `B` entornos reproduce exactamente lo que dan `B` entornos sueltos con las mismas semillas y elecciones.

**Versión en lotes.**

```python
@dataclass(frozen=True)
class BatchObs:                  # rellenada hasta max_options; B = len(self)
    state: np.ndarray            # float32 [B, state_dim]
    options: np.ndarray          # float32 [B, max_options, option_dim]; relleno en 0
    option_mask: np.ndarray      # bool    [B, max_options]; True = opción legal
    option_ids: np.ndarray       # int64   [B, max_options]; relleno -1
    option_refs: np.ndarray      # int64   [B, max_options, 2]; relleno -1
    entities: np.ndarray         # float32 [B, max_entities, entity_dim]
    entity_mask: np.ndarray      # bool    [B, max_entities]
    player: np.ndarray           # int64   [B]
    @property
    def n_options(self) -> np.ndarray          # int64 [B] = option_mask.sum(1)
    def select(self, idx: np.ndarray) -> "BatchObs"   # sub-lote por índices o máscara bool

@dataclass(frozen=True)
class BatchStep:
    obs: BatchObs                # los entornos terminados ya vienen reiniciados (auto-reset)
    rewards: np.ndarray          # float64 [B, n_players]
    done: np.ndarray             # bool [B]; True = esta transición terminó el episodio
    info: list[dict]             # info[b] es la del último paso de b; si done[b], incluye "episode_id"

class VecEnv:
    def __init__(self, make_env: Callable[[], Env], n_envs: int, seed: int = 0,
                 seed_fn: Callable[[int, int], int] | None = None): ...
    spec: EnvSpec
    n_envs: int
    def reset(self) -> BatchObs
    def step(self, choices: np.ndarray) -> BatchStep       # choices int64 [B]
```

`seed_fn(b, e)` da la semilla del episodio `e` del entorno `b`; por defecto `derive_seed(seed, b, e)` =
`int(np.random.SeedSequence([seed, b, e]).generate_state(1)[0] % 10_000_000)` (queda en el rango de
entrenamiento). La evaluación pasa `seed_fn = lambda b, e: EVAL_SEED_BASE + e*n_envs + b`, de modo que el
episodio número `i = e*n_envs + b` usa la semilla `EVAL_SEED_BASE + i` (así es pareable entre agentes).

**Decisión (revisar):** `VecEnv` es un envoltorio que apila `B` entornos individuales en un bucle de Python,
no una vectorización numpy a mano. Es la opción más simple y la única que garantiza C8 por construcción; el
costo es velocidad, que el presupuesto de la sección 2 ya asume (≈ 20–60 µs por paso de entorno).

**Política.**

```python
class Policy(Protocol):
    def act(self, obs: BatchObs) -> np.ndarray: ...   # int64 [B]; cada elemento b en [0, n_options[b])
```

Toda política es en lotes (un `Obs` suelto es un lote de 1). En juegos de dos jugadores quien llama
separa el lote por `obs.player` con `select` y consulta la política de cada jugador. Las políticas con
azar guardan su propio `np.random.Generator` (o `torch.Generator`), creado con la semilla que reciben.

**Ayudantes en `core.py`:**

```python
def derive_seed(base: int, env_idx: int, episode_idx: int) -> int
def seed_everything(seed: int) -> None
def get_device() -> torch.device                       # RLF_DEVICE, si no cuda → mps → cpu
def flatten_obs(obs: BatchObs) -> np.ndarray           # float32 [B, state_dim + max_entities*entity_dim]
@dataclass(frozen=True)
class TorchObs: ...                                    # mismos campos que BatchObs, como torch.Tensor en `device`
def to_torch(obs: BatchObs, device: torch.device) -> TorchObs
```

`flatten_obs` concatena `state` con `entities` aplanado (para MLP). Los ids y las máscaras van a `to_torch` como
`int64`/`bool`. Un `Obs` suelto se convierte a lote de 1 con `BatchObs.from_obs(obs)` (método de clase).

## 4. Entornos

Los tres son Python + numpy puro, deterministas (C1) y con versión en lotes (`VecEnv`, C8); `state_key()` está implementado en Bandit y MiniDespacho (MiniCartas lo declara no implementado). Los tres
**mantienen iguales** `state_dim`, `option_dim` y `entity_dim` entre tamaños/variantes: solo cambian
`max_options`, `max_entities`, `n_actions` y `max_steps`. Eso permite que una red entrenada en un tamaño corra
(sin reentrenar) en otro: es la demostración de las lecciones 8 y 9.

### 4.1 Bandit

Config: `Bandit(k=10, horizon=300, noise_std=1.0)`. Reglas (cada una con test en `tests/envs/test_bandit.py`):

| Id | Regla |
|---|---|
| B1 | En `reset(seed)`: `mu[i] ~ N(0, 1)` independiente para `i in range(k)`. Las medias no cambian durante el episodio. |
| B2 | Elegir el brazo `i` da recompensa `N(mu[i], noise_std²)`. |
| B3 | Todas las `k` opciones son siempre legales: `n_options == k`, `option_ids == arange(k)`. |
| B4 | El episodio dura exactamente `horizon` pasos; `done` en el paso número `horizon`. |
| B5 | `info` de cada paso contiene `mu_chosen` (media del brazo elegido) y `mu_best` (`max(mu)`), solo para calcular arrepentimiento. |

Spec: `n_players=1`, `state_dim=1` (`[t/horizon]`), `option_dim=k` (one-hot del brazo), `entity_dim=1`,
`max_options=k`, `max_entities=0` (`entities` con forma `(0, 1)`), `n_actions=k`, `max_steps=horizon`.
`option_refs` es `-1` en todo. `state_key()` es `(t,)`.

### 4.2 MiniDespacho

Un turno de despacho: llegan pedidos con ubicación y ventana horaria; hay que decidir a qué camión asignarlos o
esperar. **Ventana** = `[llegada, límite]`: el pedido es visible desde `llegada`, y llegar después de `límite` es atraso.

**Geometría y tiempo.** Cuadrícula `width × height`; depósito en `(0,0)`. Distancia = Manhattan. Un camión
tarda `distancia` ticks en llegar. Tiempo discreto `t ∈ {0, …, T}`; el episodio termina al llegar a `t = T`.

**Config y tamaños** (`DespachoConfig`; `DespachoEnv.from_size(name)`):

| Parámetro | diminuto | mediano | grande |
|---|---|---|---|
| `width × height` | 3 × 1 | 5 × 5 | 8 × 8 |
| `n_trucks` K | 2 | 3 | 6 |
| `capacity` Q (unidades por camión, para todo el episodio) | 2 | 12 | 12 |
| `max_pending` M | 1 | 4 | 12 |
| `horizon` T | 6 | 30 | 60 |
| `p_arrival` (prob. de que llegue un pedido en cada tick) | 0,5 | 0,5 | 0,8 |
| `qty_range` (demanda uniforme entera) | (1,1) | (1,3) | (1,3) |
| `slack_range` (holgura uniforme entera) | (2,2) | (4,8) | (4,10) |
| `dist_w` / `late_w` | 1 / 2 | 1 / 2 | 1 / 2 |
| `U` (costo de pedido no atendido) | 14 | 68 | 134 |

`U = dist_w·(width−1 + height−1) + late_w·T` (el peor caso de "−distancia −atraso" de un pedido); es derivado,
no un parámetro libre. `max_steps = 2·T`. Los valores de `capacity` y `p_arrival` son de diseño y pueden
ajustarse **solo** si falla el test de holgura D14 (ii) o el criterio (c) de la lección 3, anotando los números medidos en la sección 10.

**Reglas** (un test por regla en `tests/envs/test_despacho.py`):

| Id | Regla |
|---|---|
| D1 | Estado inicial: `t = 0`, todos los camiones en el depósito, con `cap = Q`, libres (`busy_until = 0`), sin pedidos pendientes. Tras inicializar, se procesan las llegadas del tick 0 (D2). |
| D2 | Llegadas del tick `t` (`t < T`): con prob. `p_arrival` llega **un** pedido: `loc` uniforme sobre todas las celdas, `qty ~ U{qty_range}`, `slack ~ U{slack_range}`, `due = t + slack`. Si hay una ranura libre entre las `M`, ocupa la de menor índice; si no, el pedido es **rechazado** y cuesta `−U` en ese momento. El calendario de llegadas de todo el episodio (qué ticks traen pedido y sus `loc`, `qty`, `slack`) se muestrea completo en `reset(seed)`; es función de la semilla y no depende de las decisiones del agente. |
| D3 | Un camión está **libre** en `t` si `busy_until ≤ t`. |
| D4 | Opción de asignar `(s, k)` (ranura `s` ocupada, camión `k`): **legal** si el camión `k` está libre **y** `cap_k ≥ qty_s`. |
| D5 | Opción **ESPERAR**: siempre legal (garantiza C2). |
| D6 | Asignar `(s, k)`: `d = |x_k−x_s| + |y_k−y_s|`; el camión pasa a `pos = loc_s`, `cap_k −= qty_s`, `busy_until = t + d`; la ranura `s` queda libre; **el tiempo no avanza** (se pueden hacer varias asignaciones en el mismo tick). |
| D7 | Recompensa de asignar: `−dist_w·d − late_w·min(T, max(0, t + d − due_s))`. Siempre `≤ 0` y `≥ −U`. |
| D8 | ESPERAR: `t += 1`; se procesan las llegadas del nuevo tick (D2) si `t < T`. Recompensa 0 (salvo rechazos de D2). |
| D9 | Fin: cuando `t == T` tras ESPERAR, `done = True`; cada pedido aún pendiente cuesta `−U` (se suma a la recompensa de ese paso). |
| D10 | Orden canónico de opciones: por ranura `s` ascendente, luego camión `k` ascendente, y ESPERAR al final. `option_id = s·K + k` para asignar y `M·K` para ESPERAR (`n_actions = M·K + 1`: 3, 13, 73). |
| D11 | Un pedido nunca se atiende dos veces; `cap_k` nunca es negativo; la suma de `qty` atendidas por camión ≤ `Q` (test de propiedad sobre 1000 episodios aleatorios). |
| D12 | `state_key()` = `(t, pend, trucks)`, con `pend` = tupla de `M` elementos, cada uno `None` o `(x, y, qty, due − t)`, y `trucks` = tupla de `K` elementos `(x, y, cap, max(0, busy_until − t))`. Se usa para tablas en "diminuto" y, en "mediano", solo para contar estados distintos (lección 5). |
| D13 | El retorno del episodio es la suma de las recompensas (sin descuento) y vale `−(distancia total·dist_w + atraso ponderado + U·no atendidos)`; `info["cost_breakdown"]` (dict con esos tres términos) los suma exactamente al retorno (test). |
| D14 | **Holgura sobre `GreedyDespacho`.** (i) "diminuto": se verifica exactamente en la lección 3, criterio (c) (política óptima − greedy, diferencia pareada LB > 0). (ii) "mediano": test en `tests/envs/test_despacho.py`: por cada semilla de evaluación se juegan 20 desarrollos de `GreedyDespacho` que en cada decisión, con prob. 0,2, eligen al azar una opción legal (el desarrollo 0 usa prob. 0, es decir greedy puro) y se toma el mejor retorno por semilla; ese mejor-en-retrospectiva − greedy, diferencia pareada, tiene LB > 0 con n = 1000 semillas. Es una cota optimista (elige con el resultado a la vista): solo prueba que desviarse de greedy puede mejorar. Si (ii) falla, se ajustan `capacity`/`p_arrival` de "mediano" hasta que pase y se anota en la sección 10. |

**Observación.** Convención de normalización: `nx = x / max(1, width−1)`, `ny = y / max(1, height−1)`,
`nd = dist / max(1, width+height−2)`.

- `state` (`state_dim = 3`): `[t/T, n_pendientes/M, n_camiones_libres/K]`.
- `entities` (`entity_dim = 6`, `max_entities = K + M`): filas `0..K−1` son camiones y filas `K..K+M−1` son las
  ranuras de pedidos (en orden de ranura); `entity_mask` es `True` en todos los camiones y en las ranuras ocupadas.
  Columnas: `[es_pedido, nx, ny, cantidad, tiempo, marca]`. Camión: `cantidad = cap/Q`,
  `tiempo = max(0, busy_until−t)/T`, `marca = 1 si libre`. Pedido: `cantidad = qty/Q`,
  `tiempo = clip((due−t)/T, −1, 1)`, `marca = 1`.
- `options` (`option_dim = 6`): `[es_esperar, nd, atraso_si_se_asigna/T, cap_restante_tras/Q, holgura/T, qty/Q]`
  con `atraso_si_se_asigna = min(T, max(0, t+d−due))`, `holgura = clip((due−t)/T, −1, 1)`. La fila de ESPERAR es
  `[1,0,0,0,0,0]`.
- `option_refs`: para asignar `(s,k)`: `(K+s, k)`; para ESPERAR: `(−1, −1)`.

`spec`: `n_players=1`, `state_dim=3`, `option_dim=6`, `entity_dim=6`, `max_options = M·K + 1`.

**MDP exacto para "diminuto"** (lección 3): `build_mdp(cfg) -> MDP` en `despacho.py`.

```python
@dataclass(frozen=True)
class MDP:
    keys: list[tuple]          # keys[s] = state_key del estado s
    index: dict[tuple, int]    # inversa de keys
    nA: int                    # = spec.n_actions
    legal: np.ndarray          # bool [nS, nA]
    R: np.ndarray              # float64 [nS, nA]; recompensa esperada inmediata (0 si no es legal)
    src: np.ndarray            # int64 [m]  ─┐ transiciones dispersas (COO): P(dst | src, act) = prob
    act: np.ndarray            # int64 [m]   │
    dst: np.ndarray            # int64 [m]   │
    prob: np.ndarray           # float64 [m]─┘
    terminal: np.ndarray       # bool [nS]; estados absorbentes (V = 0)
    p0: np.ndarray             # float64 [nS]; distribución del estado inicial
```

Se construye por búsqueda en anchura desde la distribución inicial, con la esperanza **exacta** sobre las
llegadas de D2 (llegada o no con `p_arrival`, y `loc`/`qty`/`slack` uniformes). Aserción: `nS ≤ 200_000`.
Test de fidelidad: el núcleo de dinámica expone `set_state(key)` (fuerza un estado y muestrea de nuevo el calendario de llegadas de los ticks restantes); para 20 pares `(s, a)` elegidos por semilla, la recompensa y el estado siguiente empíricos
de 20 000 pasos simulados con el entorno coinciden con `R` y `P` (frecuencia de cada `dst` dentro de ±3 errores estándar).
La coherencia `V*` (por `value_iteration`) vs retorno medio simulado de la política óptima en n = 2000 episodios
(dentro de 3 errores estándar) se prueba en `tests/agents/test_tabular.py`, no aquí (`value_iteration` aún no existe al construir el MDP).

**`GreedyDespacho`** (`agents/baselines.py`), determinista, mira solo `Obs`: entre las opciones de asignar, elige
la del pedido con menor `holgura` (columna 4 de `options`); a igualdad, la de menor `nd` (columna 1); a igualdad,
la de menor índice. Si no hay opciones de asignar, elige ESPERAR. Nunca espera si puede asignar.

### 4.3 MiniCartas

Dos jugadores, mazo de 20 cartas, mano oculta, 3 premios, una energía por turno, atacar.

**Catálogo (7 tipos de carta, 4 clases).** Clases: criatura, energía, robo, curación.

| id | nombre | clase | HP | daño | costo (energías) |
|---|---|---|---|---|---|
| 0 | Chispa | criatura | 3 | 2 | 1 |
| 1 | Lobo | criatura | 5 | 3 | 2 |
| 2 | Roca | criatura | 8 | 2 | 2 |
| 3 | Dragón | criatura | 6 | 5 | 3 |
| 4 | Energía | energía | — | — | — |
| 5 | Estudio | robo | — | — | — |
| 6 | Poción | curación | — | — | — |

**Mazos (arquetipos)**, conteo por tipo de carta `[Chispa, Lobo, Roca, Dragón, Energía, Estudio, Poción]`:

| arquetipo | conteo | total |
|---|---|---|
| `aggro` | `[6, 3, 0, 0, 8, 2, 1]` | 20 |
| `tank` | `[0, 2, 5, 0, 8, 1, 4]` | 20 |
| `dragon` | `[2, 3, 0, 3, 8, 3, 1]` | 20 |

`MAXC = [6, 3, 5, 3, 8, 3, 4]` (máximo por tipo entre arquetipos; se usa para las cotas de la lección 10).
Config: `CartasConfig(deck_p0="aggro", deck_p1="aggro")`; cada campo acepta un arquetipo o `"random"` (uno de los
tres, uniforme, con el rng del entorno).

**Reglas** (un test por regla en `tests/envs/test_cartas.py`, nombrado `test_R<n>_...`):

| Id | Regla |
|---|---|
| R1 | **Mazos.** Cada jugador tiene 20 cartas con el conteo de su arquetipo (tabla de arriba). |
| R2 | **Barajado.** Cada mazo se baraja con el rng del entorno, jugador 0 primero. Misma semilla ⇒ mismo orden. |
| R3 | **Mano inicial.** Cada jugador roba 5 cartas. Si la mano no contiene ninguna criatura, se devuelve al mazo, se baraja y se vuelve a robar 5, hasta que haya al menos una criatura (mulligan; `info["mulligans"]` lo cuenta). |
| R4 | **Premios.** Tras una mano válida, las 3 cartas superiores del mazo se apartan boca abajo como premios (ocultos para ambos jugadores). El mazo queda con 12 cartas. |
| R5 | **Colocación inicial.** Primero el jugador 0 y luego el 1 eligen una criatura de su mano como activa (fase `SETUP`, opciones `PLACE_ACTIVE`, una por tipo de criatura distinto en la mano). La criatura entra con HP máximo y 0 energías. Luego empieza el turno del jugador 0. |
| R6 | **Inicio de turno.** El jugador del turno roba 1 carta (incluido el jugador 0 en su primer turno). Si su mazo está vacío al robar, **pierde de inmediato**. |
| R7 | **Sesgo por asiento (consecuencia, no regla extra).** Como el jugador 0 roba primero, agota su mazo un turno antes que el jugador 1; R6 hace que ese desfase decida las partidas largas. Con política aleatoria en ambos asientos, `P(gana el asiento 0)` difiere de 0,5 con IC de 95 % que excluye 0,5 (n = 5000 partidas). El signo se **mide y se registra** en la sección 10; no se asume. |
| R8 | **Fase principal.** Tras robar, el jugador elige opciones hasta que juega `ATTACK` o `END_TURN`. `END_TURN` siempre es legal (garantiza C2). |
| R9 | **`PLAY_CREATURE(tipo)`.** Legal si hay esa criatura en la mano y la banca tiene menos de 2 criaturas. Entra en la primera ranura libre de la banca con HP máximo y 0 energías. |
| R10 | **`ATTACH(ranura)`.** Legal si hay una Energía en la mano y **no se adjuntó energía en este turno**; una opción por cada criatura propia en juego (activa, banca 0, banca 1). Consume la carta y suma 1 energía adjunta a esa criatura. |
| R11 | **`PLAY_DRAW`.** Legal si hay un Estudio en la mano y **no se jugó carta de apoyo (Estudio o Poción) en este turno**. Roba `min(2, cartas en el mazo)`; no pierde por robar con el mazo vacío por efecto. El Estudio va al descarte. |
| R12 | **`PLAY_HEAL(ranura)`.** Legal si hay una Poción en la mano, no se jugó carta de apoyo en este turno, y la criatura propia de esa ranura tiene HP < HP máx; una opción por criatura dañada. Cura 3 sin superar HP máx. La Poción va al descarte. |
| R13 | **`ATTACK`.** Legal si la criatura activa tiene energías adjuntas ≥ su costo. Inflige su daño a la criatura activa rival. Las energías **no se gastan**. Tras resolver, el turno termina (aunque no haya KO). |
| R14 | **KO.** Si el HP de la activa rival llega a ≤ 0: la criatura y sus energías van al descarte del rival, y el atacante toma 1 premio (la primera carta de sus premios sin tomar pasa a su mano). |
| R15 | **Victoria por premios.** Al tomar el 3.er premio, el atacante gana de inmediato. |
| R16 | **Victoria por eliminación.** Si tras un KO el rival no tiene ninguna criatura en juego (activa y banca vacías), el atacante gana de inmediato, aunque tenga menos de 3 premios. |
| R17 | **Promoción.** Si tras un KO el rival aún tiene banca, el rival debe elegir `PROMOTE(ranura de banca)` (fase `PROMOTE`; `obs.player` pasa al rival; una opción por criatura en la banca). La criatura elegida pasa a activa y su ranura de banca queda libre. Después empieza el turno del rival (R6). |
| R18 | **`END_TURN`.** Pasa el turno al rival (que roba, R6) y reinicia los indicadores del turno (energía adjuntada, apoyo jugado). |
| R19 | **Fin garantizado.** Toda partida termina en ≤ 13 turnos por jugador (el mazo de 12 se agota por R6) y en ≤ `spec.max_steps = 120` decisiones (cada carta se juega a lo sumo una vez, más un `END_TURN`/`ATTACK` por turno, más promociones). Test: 1000 partidas aleatorias respetan la cota. No existen empates. |
| R20 | **Recompensa.** Cero en todos los pasos salvo el último: ganador `+1`, perdedor `−1` (`rewards` = `[+1, −1]` o `[−1, +1]`). |
| R21 | **Sin fuga de información.** La observación de un jugador es función solo de información pública y de su propia mano/mazo. Test: para una partida a mitad de camino, cambiar en el entorno (por mutación directa de sus campos internos) la mano del rival, el orden de su mazo y el contenido de sus premios **no** cambia la `Obs` del jugador que decide. |
| R22 | **Orden y ids canónicos.** Las opciones se listan por `option_id` ascendente. Ids (`n_actions = 19`): `END_TURN=0`, `ATTACK=1`, `ATTACH(ranura j)=2+j` (j=0..2, con 0 = activa, 1..2 = banca), `PLAY_CREATURE(tipo c)=5+c` (c=0..3), `PLAY_DRAW=9`, `PLAY_HEAL(ranura j)=10+j`, `PLACE_ACTIVE(tipo c)=13+c`, `PROMOTE(banca j)=17+j` (j=0..1). |
| R23 | **Cotas de información oculta** (usadas en la lección 10). Función pública `rival_bounds(env, player) -> (lower, upper)`, dos `int64[7]`: `lower[t]` = copias del tipo `t` que el rival ha revelado: criaturas en su tablero, energías adjuntas a sus criaturas y cartas de su descarte; `upper[t] = max(0, MAXC[t] − lower[t])` = cota superior de copias del tipo `t` que aún no reveló (en su mano, mazo o premios). Test: en 1000 partidas, en todos los pasos, `lower[t] ≤ total real del rival (t)` y `(total real − lower[t]) ≤ upper[t]`: **cero violaciones**. |
| R24 | **Determinismo.** C1 se cumple incluyendo mulligans (R3) y elección aleatoria de mazo. |

**Observación** (siempre desde la perspectiva de quien decide; "propio"/"rival" son relativos a `obs.player`).

- `state` (`state_dim = 12`): `[premios_propios_restantes/3, premios_rival_restantes/3, mano_propia/20, mano_rival/20,
  mazo_propio/20, mazo_rival/20, energia_adjuntada_este_turno, apoyo_jugado_este_turno, fase_SETUP, fase_MAIN,
  fase_PROMOTE, es_asiento_0]`.
- `entities` (`entity_dim = 26`, `max_entities = 8`): filas `0..2` = criaturas propias
  `[activa, banca0, banca1]`; filas `3..5` = criaturas del rival en el mismo orden; fila `6` = resumen propio; fila
  `7` = resumen del rival. Columnas: `0..2` tipo de fila (one-hot `[tablero, resumen_propio, resumen_rival]`);
  `3` presente; `4` es_propia; `5` es_activa; `6..9` tipo de criatura (one-hot sobre ids 0..3); `10` HP/8;
  `11` energías/3; `12..18` y `19..25`: ceros en filas de tablero. **Fila 6** (resumen propio): `12..18` = conteo por
  tipo de carta en la mano propia / 8; `19..25` = conteo por tipo del resto del mazo propio + premios (lista del mazo
  menos cartas ya vistas: mano, criaturas en juego, energías adjuntas y descarte) / 8. **Fila 7** (resumen del rival): `12..18` = `lower` (R23) / 8;
  `19..25` = `upper` (R23) / 8. `entity_mask` es `True` en filas con criatura presente y en las filas 6 y 7.
- `options` (`option_dim = 19`): `0..7` tipo de opción (one-hot `[END_TURN, ATTACK, ATTACH, PLAY_CREATURE, PLAY_DRAW,
  PLAY_HEAL, PLACE_ACTIVE, PROMOTE]`); `8..14` tipo de carta involucrada (one-hot sobre los 7 tipos; Energía para
  `ATTACH`, Estudio para `PLAY_DRAW`, Poción para `PLAY_HEAL`, la criatura para `PLAY_CREATURE`/`PLACE_ACTIVE`/`PROMOTE`
  y la activa para `ATTACK`; ceros en `END_TURN`); `15..17` ranura propia objetivo (one-hot `[activa, banca0, banca1]`;
  para `PROMOTE`, la ranura de banca de origen; ceros si no aplica); `18` daño esperado del ataque / 8 (solo `ATTACK`).
- `option_refs`: `(fila de entidad de la ranura objetivo, −1)`; `ATTACK`: `(0, 3)` (activa propia, activa rival);
  `END_TURN`, `PLAY_DRAW`, `PLAY_CREATURE`, `PLACE_ACTIVE`: `(−1, −1)`.
- `state_key()`: no implementado (lanza `NotImplementedError`); MiniCartas no es tabular.

**Agente heurístico** `HeuristicCartas` (`agents/baselines.py`), determinista, mira solo `Obs`: prioridad
`PLACE_ACTIVE` (criatura de mayor daño) / `PROMOTE` (mayor energías adjuntas, luego mayor HP) en sus fases; en
`MAIN`: 1) `ATTACH` a la activa, 2) `PLAY_DRAW`, 3) `PLAY_CREATURE` (mayor daño), 4) `PLAY_HEAL` a la criatura más
dañada, 5) `ATTACK`, 6) `END_TURN`; a igualdad de criterio elige la opción de menor índice. Test de habilidad: `HeuristicCartas` le gana a `RandomPolicy` con
LB del IC de 95 % ≥ 0,80, n = 1000 partidas (500 por asiento) con mazos espejo `aggro`. Si falla, el juego no
premia la habilidad y hay que revisar las reglas antes de seguir (tarea bloqueada, no umbral reducido).

## 5. Otros módulos

### 5.1 `rlf/eval/stats.py`

```python
@dataclass(frozen=True)
class Estimate:  mean: float; lo: float; hi: float; n: int; level: float = 0.95

def mean_ci(x: np.ndarray, level: float = 0.95) -> Estimate                      # aprox. normal, ddof=1
def proportion_ci(successes: int, n: int, level: float = 0.95) -> Estimate       # Wilson
def paired_diff_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> Estimate  # mean_ci(a - b)
def normalized_score_ci(agent, rand, greedy, alpha: float, level: float = 0.95) -> Estimate  # d_i de la sección 2
def sample_size_proportion(p: float, half_width: float, level: float = 0.95) -> int   # ceil(z² p(1-p)/h²)
def sample_size_mean(sigma: float, half_width: float, level: float = 0.95) -> int     # ceil((z σ/h)²)
class InsufficientSample(ValueError): ...
def require_n(n: int, minimum: int = 1000) -> None                               # lanza InsufficientSample
def passes_lower(est: Estimate, threshold: float) -> bool                        # require_n(est.n); est.lo >= threshold
```

`z` sale de `statistics.NormalDist().inv_cdf` (sin scipy). Tests: cobertura del IC de Wilson sobre 2000
réplicas de Bernoulli(0,6) con n = 1000 cae en [93 %, 97 %]; `sample_size_proportion(0.5, 0.03) == 1068`;
`passes_lower` lanza con n = 999.

### 5.2 `rlf/eval/arena.py`

```python
def evaluate_policy(make_env, policy, n_episodes: int, n_envs: int = 100,
                    seed_base: int = EVAL_SEED_BASE) -> np.ndarray          # float64 [n_episodes]; retorno del jugador 0, ordenado por semilla

@dataclass(frozen=True)
class MatchResult:
    wins_a: int; wins_b: int; n: int
    win_rate_a: Estimate        # Wilson sobre wins_a / n
    seat0_wins: int             # partidas ganadas por el asiento 0, sumando ambos asientos de A y B

def play_games(make_env, policy_a, policy_b, n_games: int, n_envs: int = 100,
               seed_base: int = EVAL_SEED_BASE) -> MatchResult             # n_games par
def round_robin(make_env, entrants: dict[str, Policy], n_games: int, ...) -> np.ndarray  # matriz float64 [E,E] de tasas de victoria de la fila contra la columna; NaN en la diagonal
def elo_ratings(win_matrix: np.ndarray, n_games: int, anchor: int = 0, anchor_rating: float = 1000.0) -> np.ndarray
```

`play_games` (`n_games` y `n_envs` pares): la partida `i` usa la semilla `seed_base + i//2`; en las partidas pares A ocupa el asiento 0 y en las
impares el asiento 1 (misma semilla, asientos intercambiados). Lanza `AssertionError` si aparece un empate.
`elo_ratings` ajusta ratings por máxima verosimilitud de `P(i gana a j) = 1/(1+10^((r_j−r_i)/400))` (Newton o
descenso de gradiente, tolerancia 1e-6). Tests (con un entorno de juguete de 2 jugadores con probabilidad de
victoria conocida): determinismo (misma entrada ⇒ misma matriz), simetría de asientos, recuperación de
ratings verdaderos {1000, 1100, 1200} dentro de ±40 con n = 1000 partidas por par.

### 5.3 `rlf/agents/baselines.py`

`RandomPolicy(seed)`: elige uniforme entre las opciones legales. `GreedyDespacho()` y `HeuristicCartas()`: sección 4.

### 5.4 Contratos de agentes y redes (resumen; el detalle lo fijan sus tareas con TDD)

- `agents/tabular.py`: `value_iteration(mdp, gamma=1.0, tol=1e-10) -> (V, Q, policy)` con `V: float64[nS]`,
  `Q: float64[nS,nA]`, `policy: int64[nS]` (id global de acción; −1 en terminales); `QLearning(n_actions, alpha, gamma,
  eps)` con `Q` como `dict[state_key, float64[n_actions]]`, acciones ilegales enmascaradas con `option_ids`.
- `nets/mlp.py`: `MLP(in_dim, hidden, out_dim)`; `option_logits_from_ids(full_logits, obs) -> Tensor[B, max_options]`
  (toma los logits globales de las `option_ids` legales y pone `-inf` en el relleno); `MLPActorCritic`.
- `agents/dqn.py`: `DQN`, buffer de repetición, red objetivo, ε-greedy enmascarado sobre `option_ids`.
- `agents/reinforce.py`: `Reinforce(policy_net, baseline: "none"|"mean"|"value")`.
- `agents/ppo.py`: `compute_gae(rewards, values, dones, last_value, gamma, lam) -> (advantages, returns)` con
  entradas `[T, B]`; `PPO(actor_critic, cfg)` con `collect`, `update`, `train(vec_env, total_steps)`. Todo actor-crítico
  implementa `forward(obs: TorchObs) -> (logits: Tensor[B, max_options] con -inf en ilegales, value: Tensor[B])`.
  Las políticas devuelven índices de opción, nunca ids.
- `nets/pointer.py`: `PointerActorCritic`, independiente del tamaño del entorno (no aplana entidades, para poder correr en
  otro tamaño sin reentrenar): `h_global = MLP([state, media enmascarada de MLP_ent(entities)])`, `h_opciones = MLP_opt(options)`,
  `query = W_q(h_global)`, `keys = W_k(h_opciones)`, `logits = query·keys/√d`, `masked_fill(~option_mask, -inf)`; el valor sale de
  `h_global`.
- `nets/tokens.py`: `MiniPTCGNet`: tokens = 1 global + `max_entities` entidades + `max_options` opciones (cada uno
  con su proyección lineal a `d_model` y un embedding de tipo de token); `nn.TransformerEncoder`
  (`batch_first=True`, sin embedding posicional), `key_padding_mask = ~mask` (recuérdese: `True` = ignorar);
  `option_refs` suma al token de la opción el token de la entidad a la que apunta; el head es el de `pointer.py`
  usando la salida del token global como query y las salidas de los tokens de opción como keys.
  Hiperparámetros: `d_model=64`, 2 capas, 4 cabezas, `dim_feedforward=128`.

## 6. Plantilla de notebook

Fuente `notebooks/NN_nombre.py` en formato jupytext `percent`. Se genera el `.ipynb` con
`jupytext --to ipynb --set-kernel python3 notebooks/NN_nombre.py` y se ejecuta y guarda con
`pytest --nbmake --overwrite --nbmake-timeout=600 notebooks/NN_nombre.ipynb`. El `.ipynb` ejecutado se commitea
(regla de `proposal.md`). Esqueleto obligatorio (en el mismo orden que la plantilla de `proposal.md`):

```python
# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# %% [markdown]
# # NN · Título
# **Entornos:** … · **Tiempo de ejecución:** ≤ X min en CPU · **Criterio de la lección:** una frase con el umbral.

# %% [markdown]
# ## 1. La idea en una frase
# La frase. **En el sistema de Luca:** … **En un problema de negocio:** … (despacho de flota).

# %% [markdown]
# ## 2. La intuición y la fórmula
# Prosa + fórmulas en LaTeX (`$...$`). Sin demostraciones.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from rlf.core import EVAL_SEED_BASE, seed_everything
from rlf.eval.stats import mean_ci, paired_diff_ci, proportion_ci, passes_lower
TRAIN_SEED = int(os.environ.get("RLF_TRAIN_SEED", "0"))
seed_everything(TRAIN_SEED)

# %% [markdown]
# ## 3. El núcleo del algoritmo
# (código del algoritmo inline, completo y comentado; lo accesorio se importa de `rlf/`)

# %%
# …núcleo inline…  Si existe versión en rlf/, una celda posterior hace assert de que
# la versión inline y la de rlf dan el mismo resultado sobre una entrada fija.

# %% [markdown]
# ## 4. Entrenar y graficar la curva de aprendizaje

# %%
# entrenamiento con presupuesto en pasos de entorno + gráfico con IC (matplotlib)

# %% [markdown]
# ## 5. Experimentos "predice antes de correr"
# ### Experimento 1 · título
# **Predice:** pregunta concreta. Escribe tu respuesta antes de correr la celda siguiente.

# %%
# ejecuta el experimento con n >= 1000 y termina con: assert <lo que afirma la respuesta>

# %% [markdown]
# <details><summary>Respuesta esperada</summary>
#
# …respuesta con los números que mostró la celda anterior…
# </details>

# %% [markdown]
# ## 6. Criterio de la lección

# %%
# calcula cada afirmación del criterio con IC 95 % y n >= 1000, imprime la tabla y hace assert de todas

# %% [markdown]
# ## 7. Resumen y puente a la lección siguiente
```

Reglas de la plantilla: 3 a 5 experimentos por lección; la sección 6 es la misma "Criterio de la lección" del
apartado 7 de este documento; el algoritmo central de la lección aparece **inline** la primera vez y, si una
lección posterior lo necesita, se importa de `rlf/` (con la celda de verificación cruzada descrita arriba); los
gráficos usan solo matplotlib.

**Verificación de un notebook** (patrón de todas las tareas de notebook; `X` = `NN_nombre`):

```
jupytext --to ipynb --set-kernel python3 notebooks/X.py \
 && RLF_TRAIN_SEED=1 pytest --nbmake --nbmake-timeout=600 notebooks/X.ipynb \
 && pytest --nbmake --overwrite --nbmake-timeout=600 notebooks/X.ipynb
```

## 7. Lecciones

Notación: "Crea" = código de `rlf/` que la tarea escribe; "Usa" = código ya existente. Todas las afirmaciones
del criterio se evalúan con semillas `EVAL_SEED_BASE + i` disjuntas de las de entrenamiento.
`J` = retorno medio por episodio. `s(α)` = puntaje normalizado (sección 2). `V*` = valor óptimo de `value_iteration`
en "diminuto" (valor esperado del estado inicial; es negativo porque son costos).

### L0 · La interfaz agente ↔ entorno; agente aleatorio (`00_interfaz`, tres entornos)
- **Objetivo:** ver la interfaz `reset → Obs + opciones legales`, `step(índice)` y el bucle agente-entorno en los tres
  entornos, con el agente aleatorio como línea base.
- **Crea:** nada. **Usa:** `core.py`, `envs/*`, `agents/baselines.RandomPolicy`, `eval/arena.evaluate_policy`, `eval/stats`.
- **Experimentos:** (1) ¿cuántas opciones legales ve el agente en promedio en cada entorno?; (2) ¿qué pasa si se pide un
  índice fuera de rango? (`IndexError`); (3) ¿el mismo agente con distinta semilla de entorno da el mismo retorno?
- **Criterio (n = 1000 episodios por entorno, política aleatoria):**
  (a) los 1000 episodios de cada entorno (Bandit, MiniDespacho diminuto, MiniCartas) terminan sin excepción,
  con `n_options ≥ 1` en todo paso no terminal y en ≤ `spec.max_steps` pasos;
  (b) Bandit: el retorno medio por paso del aleatorio es compatible con 0,0 (chequeo de consistencia: `|media| ≤ 3 errores estándar`);
  (c) MiniCartas: 1000 partidas aleatorias sin empates (todas con `rewards` `±1`).
- **Presupuesto:** ≤ 2 min (evaluación 1 min; gráficos y experimentos 1 min).

### L1 · Bandits: explorar vs explotar (`01_bandits`, Bandit)
- **Objetivo:** entender ε-greedy y UCB, y por qué explotar sin explorar se estanca.
- **Crea:** nada (los algoritmos van inline; se ejecutan en paralelo sobre un `VecEnv` de 1000 problemas). **Usa:**
  `envs/bandit`, `core.VecEnv`, `eval/stats`.
- **Algoritmos:** estimación por promedio muestral `Q[a] += (r − Q[a]) / N[a]`; greedy puro (ε = 0),
  ε-greedy con ε ∈ {0,01, 0,1}, UCB `Q + c·sqrt(ln t / N)` con `c ∈ {1, 2}` (brazo no probado: prioridad máxima).
- **Experimentos:** (1) ¿ε = 0 o ε = 0,1 gana a 300 pasos?; (2) ¿qué c de UCB rinde más?; (3) ¿cómo cambia el
  arrepentimiento acumulado con k = 10 vs k = 50?; (4) ¿y con más ruido (`noise_std = 3`)?
- **Criterio (n = 1000 problemas, `k=10`, `horizon=300`, semillas de evaluación; valores medidos en una simulación
  de diseño con la misma especificación: aleatorio 0,003; greedy 0,997; ε=0,1 → 1,183; UCB c=1 → 1,366; óptimo medio 1,54):**
  (a) recompensa media por paso de UCB (`c = 1`): **LB ≥ 1,25**; (b) ε-greedy (ε = 0,1): **LB ≥ 1,05**; (c) ε-greedy(0,1) − greedy
  puro, diferencia pareada: **LB > 0**; (d) el aleatorio es compatible con 0,0 (chequeo de consistencia: `|media| ≤ 3 errores estándar`).
- **Presupuesto:** ≤ 3 min (6 algoritmos × 300 000 pasos de entorno ≈ 2 min; experimentos 1 min).

### L2 · Medir sin engañarse (`02_medir`, tres entornos)
- **Objetivo:** varianza, IC, cuántas partidas hacen falta, evaluación pareada, Elo y sesgo de asiento.
- **Crea:** nada (todo lo de `eval/` ya existe desde las tareas de infraestructura). **Usa:** `eval/stats`, `eval/arena`, `envs/*`,
  `agents/baselines`.
- **Experimentos:** (1) ¿cuánto varía la tasa de victoria de una moneda justa con n = 10, 100, 1000, 10000?;
  (2) ¿cuántas partidas hacen falta para distinguir 55 % de 50 %?; (3) ¿el asiento 0 de MiniCartas gana más o menos
  que el 1 con política aleatoria? (predice el signo); (4) ¿comparar por semilla pareada estrecha el IC?
- **Criterio:** (a) chequeo determinista: `sample_size_proportion(0.5, 0.03) == 1068`; (b) Elo: con 3 agentes sintéticos de ratings verdaderos
  1000/1100/1200 y n = 1000 partidas por par, los ratings recuperados por `elo_ratings` están a ≤ 40 puntos del
  verdadero (diferencias respecto al ancla); (c) sesgo de asiento: n = 5000 partidas aleatorias de MiniCartas, el IC de
  95 % de `P(gana el asiento 0)` **excluye** 0,5 (R7); (d) MiniDespacho "mediano", aleatorio vs `GreedyDespacho`, n = 1000
  semillas: la semiamplitud del IC pareado es **≤ 0,8 ×** la del IC de la diferencia sin parear.
- **Presupuesto:** ≤ 4 min (5000 partidas aleatorias ≈ 2 min; el resto 2 min).

### L3 · MDP, ecuación de Bellman, value iteration (`03_mdp_bellman`, MiniDespacho diminuto)
- **Objetivo:** ver el MDP explícito de "diminuto", escribir Bellman de óptimo y resolverlo por value iteration.
- **Crea:** `agents/tabular.value_iteration` (+ tests). **Usa:** `envs/despacho.build_mdp`, `eval/*`,
  `GreedyDespacho`. Núcleo inline en el notebook: el bucle de value iteration; una celda hace `assert` de que da el mismo
  `V` que `rlf.agents.tabular.value_iteration`.
- **Experimentos:** (1) ¿cuántas barridas necesita para converger con γ = 1 y horizonte 6?; (2) ¿cambia la política si
  `late_w` pasa de 2 a 0?; (3) ¿cuánto vale esperar en el estado inicial?; (4) ¿qué estado alcanzable tiene el mayor
  valor absoluto?
- **Criterio (n = 2000 episodios, γ = 1):** (a) chequeo determinista: residuo de Bellman `max_s,a |Q(s,a) − (R + Σ P V)|` **< 1e-9**;
  (b) chequeo de consistencia: el retorno medio de la política óptima simulada en el entorno está a **≤ 3 errores estándar de `V*`**;
  (c) política óptima − `GreedyDespacho`, diferencia pareada: **LB > 0**.
- **Presupuesto:** ≤ 2 min (construir MDP < 30 s; simulaciones 1 min).

### L4 · Q-learning tabular (`04_qlearning`, MiniDespacho diminuto)
- **Objetivo:** aprender sin conocer el modelo; efecto de γ y ε.
- **Crea:** `agents/tabular.QLearning` (+ smoke test de aprendizaje). **Usa:** `value_iteration` para `V*`,
  `envs/despacho`. Núcleo inline: la actualización `Q[s,a] += α (r + γ max_a' Q[s',a'] − Q[s,a])` y el bucle.
- **Experimentos:** (1) ε constante 0,3 vs decreciente; (2) γ = 0,5 vs 0,9 vs 1,0 evaluados en retorno **sin descontar**;
  (3) ¿cuántos episodios hacen falta para llegar al 95 %?; (4) ¿qué fracción de los estados alcanzables se visitó?
- **Criterio (n = 2000 episodios de evaluación, política greedy ε = 0, γ = 1):** (a) política aprendida − política óptima
  de `value_iteration`, diferencia pareada por semilla: **LB ≥ −0,05·|V\*|** (≥ 95 % del valor óptimo).
- **Presupuesto:** ≤ 6 min (entrenamiento ≈ 300 000 episodios × 6 pasos ≈ 5 min; el resto 1 min). El número exacto de episodios
  lo fija la tarea del notebook con la regla de la sección 2.

### L5 · De tabla a red neuronal (`05_dqn`, MiniDespacho)
- **Objetivo:** por qué una tabla no escala y cómo una red generaliza (DQN simple).
- **Crea:** `nets/mlp.py` (`MLP`, `option_logits_from_ids`, `MLPActorCritic`), `agents/dqn.py`. **Usa:**
  `core.flatten_obs`, `envs/despacho`, `value_iteration`, `GreedyDespacho`. Núcleo inline: pérdida TD con red objetivo y bucle de repetición.
- **Experimentos:** (1) ¿cuántos estados distintos ve una política aleatoria en "mediano" vs "diminuto"?; (2) DQN con y
  sin red objetivo; (3) tamaño del buffer; (4) ¿la red generaliza a estados no vistos en "diminuto"?
- **Criterio (n = 1000 episodios, semillas de evaluación):** (a) "diminuto": DQN − política óptima, diferencia pareada:
  **LB ≥ −0,10·|V\*|**; (b) "mediano": **`s(0,5)` ≥ 0,5** (DQN a mitad de camino entre aleatorio y `GreedyDespacho`);
  (c) "mediano": en 1000 episodios aleatorios, `estados distintos (state_key) / pasos totales ≥ 0,7` (chequeo determinista sobre esos episodios; por qué no cabe en una tabla).
- **Presupuesto:** ≤ 7 min (DQN "diminuto" 2 min, DQN "mediano" 4 min, evaluación 1 min).

### L6 · Policy gradient: REINFORCE con baseline (`06_reinforce`, MiniDespacho mediano)
- **Objetivo:** optimizar la política directamente; por qué el baseline reduce la varianza.
- **Crea:** `agents/reinforce.py`. **Usa:** `nets/mlp.MLPActorCritic`, `envs/despacho`, `GreedyDespacho`.
  Núcleo inline: `loss = −Σ log π(a|s)·(G_t − b(s))`.
- **Experimentos:** (1) sin baseline vs baseline de media vs baseline aprendido (varianza del gradiente sobre 1000
  episodios); (2) tasa de aprendizaje ×10; (3) entropía de la política durante el entrenamiento; (4) ¿γ < 1 ayuda?
- **Criterio (n = 1000 episodios):** (a) REINFORCE con baseline aprendido: **`s(0,5)` ≥ 0,5**; (b) con la política final
  fija y n = 1000 episodios, la varianza total de los gradientes por episodio (suma sobre parámetros de la varianza entre
  episodios de cada componente del gradiente de `Σ_t log π(a_t|s_t)·(G_t − b)`) con baseline aprendido es **menor** que con
  `b = 0` (chequeo determinista: cociente < 1).
- **Presupuesto:** ≤ 6 min (entrenamiento 5 min; evaluación y experimentos 1 min).

### L7 · Actor-critic, GAE y PPO (`07_ppo`, MiniDespacho mediano)
- **Objetivo:** valor como crítico, ventaja generalizada y actualización con recorte.
- **Crea:** `agents/ppo.py` (`compute_gae`, `PPO`, `PPOConfig`). **Usa:** `MLPActorCritic`, `VecEnv`, `GreedyDespacho`.
  Núcleo inline: `compute_gae` y la pérdida recortada; una celda hace `assert` de que la versión inline y la de `rlf` coinciden.
- **Experimentos:** (1) `λ = 0` vs `0,95` vs `1`; (2) con y sin recorte (`clip_eps` ∈ {0,1, 0,2, ∞}); (3) 1 vs 4
  épocas por lote; (4) PPO vs REINFORCE con el mismo número de pasos.
- **Criterio (n = 1000 semillas de evaluación):** (a) PPO le gana a `GreedyDespacho`: diferencia pareada **LB > 0**
  (equivale a `s(1)` con margen). El test unitario `tests/agents/test_ppo.py::test_gae_hand_computed` compara GAE con un
  cálculo hecho a mano (ejemplo de 3 pasos con γ = 0,9 y λ = 0,8) y `::test_ppo_learns_smoke` verifica que PPO mejora al
  aleatorio en "diminuto" en 20 000 pasos.
- **Presupuesto:** ≤ 8 min (entrenamiento 300 000 pasos ≈ 6 min; evaluación y experimentos 2 min, con presupuestos reducidos
  en los experimentos: 60 000 pasos cada uno, n = 1000).

### L8 · Acciones variables: masking + pointer (`08_masking_pointer`, MiniDespacho y MiniCartas)
- **Objetivo:** dos formas de manejar opciones legales variables: máscara sobre ids fijos, y pointer attention sobre
  features de opciones (la que usa Luca); y por qué la segunda transfiere entre tamaños.
- **Crea:** `nets/pointer.py` (`PointerActorCritic`) y soporte en `agents/ppo.py` para cualquier `ActorCritic`. **Usa:**
  `option_logits_from_ids` (creada en la lección 5), `envs/*`, `agents/baselines`. Núcleo inline: el cálculo `query·keys/√d` con
  máscara.
- **Experimentos:** (1) ¿qué pasa sin máscara (probabilidad sobre ilegales)?; (2) máscara con id fijo vs pointer en
  "mediano"; (3) ¿un modelo con máscara sobre ids puede correr en "grande"? (no: distinto `n_actions`); (4) ¿el pointer sí?
- **Criterio:** (a) máscara: en 1000 episodios en cada uno de "mediano" y MiniCartas con una red **sin entrenar**, la
  probabilidad total sobre opciones ilegales es exactamente 0 y el índice elegido siempre `< n_options` (0 violaciones);
  (b) "mediano", pointer entrenado con PPO: pointer − `GreedyDespacho`, diferencia pareada **LB ≥ −0,02·|J_greedy|**
  (paridad con greedy); (c) transferencia: el mismo pointer, **sin reentrenar**, en "grande": pointer − aleatorio,
  diferencia pareada **LB > 0** (n = 1000); (d) MiniCartas: pointer entrenado contra `RandomPolicy`, tasa de victoria
  **LB ≥ 0,70** (n = 1000 partidas, 500 por asiento, mazos espejo `aggro`).
- **Presupuesto:** ≤ 8 min (pointer "mediano" 3 min, pointer MiniCartas 3 min, evaluaciones y experimentos 2 min).

### L9 · Entidades como tokens + Transformer (`09_transformer`, MiniDespacho y MiniCartas)
- **Objetivo:** construir el mini-PTCGNet: tokens global + entidades + opciones, atención, y pointer sobre las salidas.
- **Crea:** `nets/tokens.py` (`MiniPTCGNet`). **Usa:** `nets/pointer`, `agents/ppo`, `envs/*`. Núcleo inline: un bloque de
  autoatención con máscara escrito a mano (una celda lo compara con `nn.MultiheadAttention`, máxima diferencia ≤ 1e-5)
  y el ensamblado de tokens.
- **Experimentos:** (1) ¿cambia la salida al permutar las entidades?; (2) ¿y sin `option_refs`?; (3) ¿cuántos parámetros
  tiene y cuánto tarda un paso frente al MLP?; (4) ¿ve la red las filas de relleno? (no).
- **Criterio:** (a) chequeo determinista de invarianza a permutaciones: en n = 1000 observaciones (500 de cada entorno) con las entidades permutadas
  junto con sus referencias, los logits cambian **≤ 1e-5**; (b) chequeo determinista: parámetros de `MiniPTCGNet` **≤ 150 000**; (c) "mediano":
  **`s(0,5)` ≥ 0,5** (n = 1000); (d) el mismo modelo en "grande" sin reentrenar: transformer − aleatorio, diferencia pareada
  **LB > 0**; (e) MiniCartas contra `RandomPolicy`: tasa de victoria **LB ≥ 0,80** (n = 1000, 500 por asiento).
- **Presupuesto:** ≤ 8 min (entrenar "mediano" 3 min, MiniCartas 3,5 min, evaluación y experimentos 1,5 min).

### L10 · Dos jugadores: self-play e información oculta (`10_selfplay`, MiniCartas)
- **Objetivo:** self-play con PPO y recompensa +1/0/−1; manejar información oculta con hechos y cotas demostrables, no
  con priors.
- **Crea:** `agents/ppo.train_selfplay` (el mismo modelo controla ambos asientos, oponente muestreado del historial
  reciente) y `envs/cartas.rival_bounds` (R23; ya existe desde el entorno MiniCartas y esta lección lo usa). **Usa:** `MiniPTCGNet`, `arena.play_games`.
- **Experimentos:** (1) ¿self-play mejora de forma monótona la tasa de victoria contra `RandomPolicy`?; (2) ¿qué
  fracción de las cartas ocultas del rival queda determinada por las cotas a mitad de partida?; (3) un "prior" (conteo
  promedio de los tres arquetipos menos lo ya revelado) vs las cotas: fracción de pasos en que el prior estima un conteo
  negativo (imposible) y error medio frente al conteo real de cartas no reveladas; (4) ¿el sesgo de asiento cambia tras entrenar?
- **Criterio (n = 1000 partidas, 500 por asiento, mazos espejo `aggro`):** (a) el agente de self-play le gana a
  `RandomPolicy` con tasa **LB ≥ 0,90**; (b) chequeo determinista: **0 violaciones** de R23 en 1000 partidas jugadas por el agente
  entrenado (en todos los pasos).
- **Presupuesto:** ≤ 8 min (self-play 6 min ≈ 400 000 pasos; evaluación y experimentos 2 min).

### L11 · Liga de oponentes, arena, matriz de counters, selección (`11_liga_arena`, MiniCartas)
- **Objetivo:** organizar una liga balanceada por arquetipo, una arena con n fijo por par, ver la matriz de counters y
  elegir qué subir.
- **Crea:** en `eval/arena.py`: `League` (`add(name, policy, archetype)`, `sample(rng) -> name` balanceado por
  arquetipo: primero el arquetipo uniforme, luego una entrada uniforme dentro del arquetipo), `counter_matrix(...)`
  (envoltura de `round_robin` con IC por celda) y `select_maximin(win_matrix) -> int` (entrada con mayor
  `min_j win_rate[i,j]`). **Usa:** `round_robin`, `elo_ratings`, `HeuristicCartas`, `agents/ppo`.
- **Liga:** 6 entradas = {`HeuristicCartas`, PPO especialista de 90 s} × {`aggro`, `tank`, `dragon`}; arquetipo = mazo.
  Cada especialista se entrena contra la mezcla de las entradas heurísticas.
- **Experimentos:** (1) ¿el ranking por Elo coincide con el ranking por maximin?; (2) ¿la matriz tiene ciclos (A gana a B,
  B a C, C a A)?; (3) ¿cuántas partidas por par hacen falta para que el orden no cambie?; (4) muestreo balanceado vs uniforme
  sobre entradas.
- **Criterio:** (a) determinismo: dos corridas de `round_robin` con las mismas semillas dan matrices idénticas; (b)
  muestreo balanceado: con una liga de 5/1/1 entradas por arquetipo, en 30 000 muestras cada arquetipo aparece con
  frecuencia **1/3 ± 0,02**; (c) matriz sintética de piedra-papel-tijera: los ratings Elo difieren **≤ 20** puntos mientras
  cada entrada tiene un rival que le gana con tasa ≥ 0,70; (d) liga real, 15 pares × n = 1000 partidas (500 por asiento):
  **≥ 3 pares** con IC de 95 % (Wilson) que excluye 0,5.
- **Presupuesto:** ≤ 8 min (3 especialistas × 90 s = 4,5 min; 15 000 partidas por lotes 2,5 min; experimentos 1 min).

### L12 · Currículo generalista → especialista y negocio (`12_curriculo_negocio`, MiniDespacho y MiniCartas)
- **Objetivo:** entrenar primero un generalista y afinar un especialista; llevar el pipeline a un problema de negocio:
  simulador, brecha sim-to-real y explotación de errores del simulador.
- **Crea:** en `envs/despacho.py`: perfiles de demanda (`PROFILES = {"urgente", "holgado", "pesado"}`, ver abajo) y la
  variante con error `bug="late_at_dispatch"`. **Usa:** `PPO`, `MiniPTCGNet` o `MLPActorCritic` (el que menos tarde),
  `GreedyDespacho`, `HeuristicCartas`.
- **Perfiles** (sobre "mediano"): `urgente` = `p_arrival 0,6`, `slack_range (2,4)`; `holgado` = `p_arrival 0,25`,
  `slack_range (5,9)`; `pesado` = `qty_range (2,3)`, `capacity 15`. **Error del simulador** `late_at_dispatch`: D7 mide el
  atraso con `max(0, t − due)` en lugar de `max(0, t + d − due)` (el atraso se cobra al despachar, no al llegar); es un
  error realista (medir en el momento equivocado) que premia mandar camiones lejanos sin costo de atraso.
- **Experimentos:** (1) generalista vs especialista en `urgente`; (2) el especialista evaluado en un perfil que no es el suyo;
  (3) ¿cuánto sube la recompensa reportada en el simulador con error frente al simulador correcto?; (4) MiniCartas:
  generalista (mazos aleatorios) vs especialista `aggro`, sin criterio, presupuestos de 60 s y 40 s.
- **Criterio (n = 1000 semillas de evaluación en el simulador correcto):** (a) en el perfil `urgente`, especialista −
  generalista, diferencia pareada **LB > 0**; (b) agente afinado en el simulador con error − especialista (ambos evaluados
  en el simulador correcto) : diferencia pareada **UB < 0** (la política que explota el error rinde peor en la realidad).
  Ambos parten del mismo generalista y se afinan con el mismo presupuesto.
- **Presupuesto:** ≤ 8 min (generalista 2,5 min, especialista 1 min, agente con error 1 min, parte MiniCartas 1,7 min,
  evaluaciones 1,8 min).

## 8. Definición de "tarea terminada"

1. Tests escritos primero y vistos fallar; luego implementación (TDD, regla de `proposal.md`).
2. La `verification` de la tarea pasa y su salida se pega en el mensaje del commit.
3. Tareas de notebook: el `.ipynb` queda commiteado **ejecutado**, con salidas y gráficos; la celda "Criterio de la
   lección" pasa con `RLF_TRAIN_SEED=0` y `=1`; el tiempo medido es ≤ 10 min (objetivo ≤ 8).
4. Un commit por tarea (`tipo(área): descripción`, en español, sin `Co-Authored-By` ni menciones de IA), con la
   línea de `tasks.md` marcada `[x]` en el mismo commit; push a `main`.

## 9. Decisiones de diseño que Christian debería revisar primero

1. **`VecEnv` como envoltorio de Python** (no numpy vectorizado). Simple y correcto por construcción; más lento. Si
   los notebooks no caben en 8 min habrá que optimizar entornos concretos.
2. **MiniDespacho "mediano/grande" elige pedido y camión** (opciones = pares `(pedido, camión)` + esperar), mientras que
   `proposal.md` dice "asigna el pedido pendiente a un camión legal": en "diminuto" coincide exactamente (`M = 1`); la
   generalización a `M > 1` es lo que da opciones variables y entidades como tokens.
3. **Capacidad por episodio sin recarga** y costo `U` para pedidos no atendidos o rechazados (= el peor "−distancia −atraso").
   Sin `U` el agente ganaría ignorando pedidos.
4. **Sesgo de asiento emergente** (R7): viene del orden de turnos y del mazo que se agota (R6), no de una regla de
   compensación. Signo por medir. Alternativa: una regla explícita.
5. **MiniCartas: reglas propias** (7 cartas, 3 arquetipos, premios ocultos, mulligan, pérdida por mazo agotado,
   sin retirada ni empates). Los tres arquetipos no están calibrados como piedra-papel-tijera; la matriz de la lección 11
   mostrará lo que salga.
6. **Umbrales de lección 1 medidos por simulación; el resto son umbrales relativos** (contra `V*`, contra greedy o
   contra aleatorio) porque no se pueden medir sin implementar. Si algún umbral no se alcanza en 3 intentos, la tarea
   se bloquea y decide Christian.
7. **Altas a la estructura del repo:** `agents/baselines.py` (aleatorio, greedy, heurística), `build/` ignorado,
   `tests/smoke_notebook.py`.
8. **Criterio con dos semillas de entrenamiento** (0 y 1): duplica el tiempo de verificación de cada notebook.
9. **Torch de PyPI en Linux es la variante CUDA** (descarga grande) porque el índice CPU no es alcanzable desde
   el sandbox; si el disco de una sesión cloud no alcanza, hay que revisar.
10. **La lección 12 combina cuatro entrenamientos** en ≤ 8 min; es la más ajustada de tiempo.

## 10. Registro de calibración

Vacío al diseñar. Quien implemente añade aquí, con fecha, los números medidos que justifiquen cualquier ajuste
permitido por este documento (`capacity`/`p_arrival` en D14; pasos de entrenamiento por notebook; signo y magnitud del sesgo
de asiento de R7). Un umbral de criterio nunca se edita aquí: solo Christian lo cambia.

- 2026-09-29 · T8 · D14 (ii) en "mediano" **sin ajuste** de `capacity`/`p_arrival`: mejor-en-retrospectiva − greedy
  = +19,995 (IC 95 % [17,79; 22,20], n = 1000); retorno medio de greedy −77,47. MDP exacto de "diminuto":
  2427 estados (91 terminales), 6020 transiciones no nulas (límite 200 000).
- 2026-09-29 · T12 · Sesgo de asiento R7 (MiniCartas, mazos espejo `aggro`, política aleatoria en ambos asientos,
  n = 5000): `P(gana el asiento 0)` = 0,397 (1987/5000; IC 95 % [0,384; 0,411]). **Signo: el asiento 0 pierde más**
  (agota su mazo un turno antes, R6). El IC excluye 0,5; el test fija el signo (`hi < 0,5`).
- 2026-09-29 · T12 · Habilidad: `HeuristicCartas` vs `RandomPolicy`, mazos espejo `aggro`, n = 1000 (500 por asiento):
  tasa de victoria 0,964 (IC 95 % [0,951; 0,974]); umbral LB ≥ 0,80 superado sin ajustes.
