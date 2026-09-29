# Curso de fundamentos de RL — brief aprobado

Brief de diseño aprobado por Christian el 2026-09-29. Es la fuente de verdad para escribir
`design.md` y `tasks.md` de este change. Quien implemente no vio la conversación de diseño: todo lo
decidido está aquí.

## Por qué

Christian quiere entender paso a paso cómo funcionan los sistemas de aprendizaje por refuerzo del
estilo del ganador de la Kaggle PTCG AI Battle Challenge (Simulation track, 1º lugar, "Luca"):

- un Transformer chico de 2,24M parámetros que representa el estado como tokens (un token global,
  cartas en juego, resúmenes de cartas propias y del rival, y un token por opción legal);
- un head de acción por pointer attention: la query sale del token global, las keys de los tokens
  de opción, se enmascaran las ilegales y se aplica softmax;
- entrenamiento con PPO + GAE en self-play, con recompensa pura +1/0/−1;
- currículo de generalista a especialista, liga de oponentes balanceada por arquetipo, y una arena
  local con n fijo por par para elegir qué subir;
- información oculta modelada solo con hechos y cotas demostrables, sin priors bayesianos.

Referencia pública: https://www.kaggle.com/w/103970 (writeup "Aim to Be a Pokémon Master").

## Objetivo y criterio de éxito

Un curso de **fundamentos de RL**, aplicable tanto a **juegos de cartas** como a **problemas de
negocio** (despacho o ruteo de flota). Al terminarlo, Christian puede explicar y modificar cada
pieza del pipeline de Luca y aplicarla a un problema propio.

## Formato (decidido)

- **Notebooks progresivos**, uno por lección, en **español**. El código de cada lección viene
  **completo y funcionando**, explicado, y termina con **experimentos guiados** de "predice antes
  de correr" (3 a 5 por lección, con la respuesta esperada en un bloque plegado).
- Intuición y fórmulas, sin demostraciones formales.
- La fuente de cada notebook es un `.py` en formato jupytext percent (`notebooks/NN_nombre.py`). El
  `.ipynb` se genera desde ahí y se commitea **ejecutado**, con sus salidas, para poder leerlo en
  GitHub.

Plantilla de cada notebook:

1. La idea en una frase, y dónde aparece en el sistema de Luca y en un problema de negocio.
2. La intuición y la fórmula.
3. El núcleo del algoritmo inline, importando lo accesorio desde `rlf/`.
4. Entrenar y graficar la curva de aprendizaje.
5. Experimentos "predice antes de correr".
6. Resumen y puente a la lección siguiente.

## Enfoque (decidido): escalera de conceptos sobre entornos propios

Desde la lección 0 se usa una sola interfaz, copiada del motor del PTCG: el entorno entrega una
observación y la **lista de opciones legales**, y el agente devuelve **índices** de opciones. Todos
los entornos del curso la implementan.

### Entornos (Python + numpy puro, con semilla, deterministas y con versión en lotes)

- **Bandit**: k brazos con recompensa ruidosa (elegir proveedor, o carta inicial).
- **MiniDespacho**: K camiones con capacidad y pedidos con ubicación y ventana horaria. En cada
  decisión se asigna el pedido pendiente a un camión legal, o se espera. Recompensa: −distancia
  −atraso. Configurable desde "diminuto" (cabe en una tabla Q) hasta "grande" (necesita tokens).
- **MiniCartas**: dos jugadores, mazo de 20 cartas de 4 tipos (criatura con HP y ataque, energía,
  robo, curación), mano oculta, 3 premios, una energía por turno, atacar. Tiene acciones legales
  variables, información oculta y sesgo por asiento. Las reglas completas van como tabla en
  `design.md`, con un test por regla.

### Currículo (13 notebooks)

| # | Concepto | Entorno |
|---|---|---|
| 0 | La interfaz agente ↔ entorno; agente aleatorio | los tres |
| 1 | Bandits: explorar vs explotar (ε-greedy, UCB) | Bandit |
| 2 | Medir sin engañarse: varianza, intervalos de confianza, cuántas partidas hacen falta, Elo | los tres |
| 3 | MDP, ecuación de Bellman, value iteration (modelo conocido) | MiniDespacho diminuto |
| 4 | Q-learning tabular (sin modelo); efecto de γ y ε | MiniDespacho diminuto |
| 5 | De tabla a red neuronal: por qué y cómo (DQN simple) | MiniDespacho |
| 6 | Policy gradient: REINFORCE con baseline | MiniDespacho |
| 7 | Actor-critic, GAE y PPO | MiniDespacho |
| 8 | Acciones variables: masking + acción como features (pointer) | MiniDespacho y MiniCartas |
| 9 | Entidades como tokens + Transformer (mini-PTCGNet) | MiniDespacho y MiniCartas |
| 10 | Dos jugadores: self-play e información oculta (hechos y cotas vs priors) | MiniCartas |
| 11 | Liga de oponentes, arena, matriz de counters, selección | MiniCartas |
| 12 | Currículo generalista → especialista, y cómo llevarlo a negocio (simulador, sim-to-real, exploits del simulador) | ambos |

## Estructura del repo

```
rlf/                  # librería que el curso va construyendo
  core.py             # interfaz: reset() → obs + opciones legales; step(índices)
  envs/               # bandit.py · despacho.py · cartas.py
  agents/             # tabular · dqn · reinforce · ppo
  nets/               # mlp · pointer · tokens (transformer)
  eval/               # stats (IC, tamaño de muestra) · arena (round-robin, Elo)
notebooks/            # NN_nombre.py (fuente jupytext) + NN_nombre.ipynb ejecutado
tests/
openspec/changes/curso-rl/  # proposal (este archivo) · design · tasks
```

## Stack

Python 3.12, PyTorch (dispositivo automático: cuda → mps → cpu), numpy, matplotlib, jupytext,
pytest, nbmake. Se eligió PyTorch sobre JAX por la cantidad de material de aprendizaje y su soporte
en Mac. **Antes de escribir código contra una librería, consultar su documentación actual** para la
versión que se instale, nunca de memoria.

## Restricciones

- Cada notebook ejecuta de punta a punta en **≤ 10 min en CPU**: las sesiones cloud corren en CPU
  Linux, y Christian usa un Apple M5 de 16 GB.
- **Toda afirmación de aprendizaje se mide con n ≥ 1000** episodios o partidas, y se reporta con
  intervalo de confianza del 95 %. Nunca se concluye a partir de una sola corrida.
- Git: autor `Christian Elias Rangel Intriago <christian.rangel@cloud.uautonoma.cl>`. **Sin
  trailers `Co-Authored-By` ni menciones de IA** en commits, código ni docs. Commits
  `tipo(área): descripción` en español (`feat`, `test`, `docs`, `chore`). Un commit por tarea
  verificada.
- Identificadores y comentarios de código en inglés; prosa de notebooks y docs en español.
- Sin features no pedidas; tres líneas parecidas son mejores que una abstracción prematura.

## Verificación (una tarea se cierra solo con evidencia ejecutada)

- `pytest` cubre:
  - las reglas de cada entorno;
  - que el masking nunca elige una acción ilegal;
  - GAE comparado contra un cálculo hecho a mano;
  - un smoke test de aprendizaje por agente.
- `pytest --nbmake notebooks/NN_*.ipynb` ejecuta el notebook completo.
- **Criterio de aprendizaje por lección**, con umbral numérico y n ≥ 1000. Ejemplos:
  - Q-learning alcanza ≥ 95 % del valor óptimo que calcula value iteration;
  - PPO le gana a la heurística greedy de despacho;
  - el agente de self-play le gana ≥ 90 % al aleatorio.

## Ejecución

Sesiones de Claude Code en la nube (claude.ai/code) sobre este repo, en Sonnet. La cola es
`tasks.md`, con este formato de línea:

`- [ ] T<n> — <título> · files: <rutas> · verification: <comando exacto>`

Cada sesión:

1. Toma la primera tarea `[ ]` sin `bloqueada`.
2. Hace TDD: escribe el test, verifica que falla y recién entonces implementa.
3. Corre la `verification` de la tarea y pega la salida en el commit o en `tasks.md`.
4. Marca la tarea `[x]` y hace commit y push a `main`.

Si una tarea no pasa tras 3 intentos con hipótesis distintas, se marca `· bloqueada: <motivo>` y la
sesión sigue con otra. **La primera sesión solo escribe `design.md` y `tasks.md` y se detiene**,
para que Christian los revise antes de implementar nada.
