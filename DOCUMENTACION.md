# Explicación Detallada y Arquitectura del Repositorio

> **Proyecto:** Heterogeneous Evolution Strategies for Mixed-Variable Optimization (TFG)  
> **Propósito:** Documento explicativo integral de la estructura, arquitectura evolutiva, flujo de ejecución y kernels matemáticos del repositorio.

---

## 1. Visión General del Proyecto

Este repositorio contiene el prototipo y marco experimental de una **Estrategia de Evolución Heterogénea (Heterogeneous Evolution Strategy - ES)** diseñada para la **optimización de variables mixtas**.

A diferencia de las Estrategias de Evolución convencionales (orientadas únicamente a variables continuas en $\mathbb{R}^n$), este sistema permite co-evolucionar **en un único genoma** 5 tipos distintos de variables con sus respectivos parámetros de estrategia adaptativos:

1. **Variables Reales ($\mathbb{R}$):** Continuas, mutadas mediante perturbación Gaussiana.
2. **Variables de Conteo / Enteras ($\mathbb{Z}$):** Discretas, mutadas mediante la distribución de Skellam (diferencia de dos distribuciones de Poisson).
3. **Variables Binarias ($\mathbb{B}$):** Flip de bits independiente según probabilidad adaptativa mediante transformación Logit-Sigmoide.
4. **Variables Categóricas ($\mathbb{C}$):** Re-muestreo uniforme sobre categorías alternativas excluyendo el estado actual.
5. **Variables Difusas / Fuzzy ($\mathbb{F}$):** Tripletas de puntos de control $(a, b, c)$ con restricción monótona $a \le b \le c$ y reflejo especular en $[0, 1]$.

El repositorio incluye además un **entorno de evaluación comparativa (Benchmark Framework)** contra optimizadores de referencia (*Isotropic ES*, *sep-CMA-ES* y *Standard CMA-ES*).

---

## 2. Mapa del Repositorio (Gráfico en Texto Plano)

```text
TFG_raona/
│
├── README.md                              # Guía rápida de instalación y ejecución
├── requirements.txt                       # Dependencias del proyecto (numpy, scipy, cma, matplotlib...)
├── DOCUMENTACION.md                       # Este documento explícito de arquitectura y flujo
│
├── docs/                                  # Documentación técnica detallada
│   ├── arquitectura-es.md                 # Diseño del motor evolutivo heterogéneo
│   ├── benchmarks.md                      # Descripción de funciones objetivo y suite de evaluación
│   └── configuracion.md                   # Explicación de hiperparámetros de configuración
│
├── openspec/                              # Especificaciones de diseño OpenSpec
│
├── src/                                   # Código fuente principal de la aplicación
│   ├── es_model/                          #   [MOTOR EVOLUTIVO HETEROGÉNEO]
│   │   ├── config.py                      #     Variables globales, dimensiones, bounds e hiperparámetros
│   │   ├── kernels.py                     #     Kernels matemáticos vectorizados de adaptación y mutación
│   │   ├── engine.py                      #     Estructura Individual, clonación, mutación y selección
│   │   └── main.py                        #     Entrypoint CLI / Menú interactivo / Bucle de optimización
│   │
│   ├── benchmarks/                        #   [FUNCIONES OBJETIVO Y BENCHMARKS]
│   │   └── functions.py                   #     Funciones de prueba (Sphere, Rastrigin, Rosenbrock, Ackley)
│   │
│   └── runners/                           #   [MARCO EXPERIMENTAL / ALGORITMOS COMPARATIVOS]
│       ├── custom_es.py                   #     Wrapper de la ES Heterogénea (Prototype 1)
│       ├── isotropic_es.py                #     Wrapper de ES Isotrópica (sigma global único)
│       ├── one_plus_one_es.py             #     Wrapper de (1+1)-ES
│       ├── sep_cma_es.py                  #     Wrapper de Separable CMA-ES
│       └── standard_cma_es.py             #     Wrapper de Standard CMA-ES (Límite superior)
│
└── experiments/                           # Experimentos y artefactos analíticos
    └── notebooks/
        └── experiment_1.ipynb             # Notebook con gráficas de convergencia
```

---

## 3. Diagramas de Flujo y Arquitectura (Texto Plano)

### Diagrama 3.1: Flujo Global de Ejecución (`main.py`)

```text
                     +-------------------------------+
                     |     Inicio: python -m         |
                     |     src.es_model.main         |
                     +---------------+---------------+
                                     |
                                     v
                     +-------------------------------+
                     |  ¿Pasa argumento CLI directo? |
                     +---------------+---------------+
                                    / \
                              Sí   /   \  No
                                  /     \
                                 v       v
         +-----------------------+     +-------------------------------+
         | Ejecuta el modo       |     |  Muestra Menú Interactivo    |
         | especificado (CLI)    |     |  1) model     2) benchmarks   |
         +-----------+-----------+     |  3) smoke     4) all  5) exit |
                     |                 +---------------+---------------+
                     |                                 | Selección del Usuario
                     +----------------+----------------+
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
       v                              v                              v
[ MODO MODEL ]                [ MODO BENCHMARKS ]             [ MODO SMOKE / ALL ]
Ejecuta la ES                 Ejecuta la suite comparativa   Ejecuta validaciones
Heterogénea con               contra Isotropic ES,           rápidas o la combinación
parámetros de config.py       sep-CMA-ES y Standard CMA-ES   de model + benchmarks
       |                              |                              |
       +------------------------------+------------------------------+
                                      |
                                      v
                     +-------------------------------+
                     | Imprime métricas/resultados   |
                     |  y finaliza la ejecución      |
                     +-------------------------------+
```

---

### Diagrama 3.2: Estructura de un Individuo (`Individual`)

Un `Individual` co-evoluciona sus **variables de decisión** junto con sus **parámetros de estrategia**:

```text
+-----------------------------------------------------------------------------------------+
|                                  ESTRUCTURA INDIVIDUAL                                  |
+-----------------------------------------------------------------------------------------+
|  VARIABLES DE DECISIÓN (Genome)               |  PARÁMETROS DE ESTRATEGIA (Self-Adapt)  |
+-----------------------------------------------+-----------------------------------------+
|  1. real (float64[N_REAL])                    |  sigma_real (float64[N_REAL])           |
|  2. count (int64[N_COUNT])                    |  lambda_count (float64[N_COUNT])        |
|  3. binary (uint8[N_BINARY])                  |  logit_binary (float64[N_BINARY])       |
|  4. categorical (int64[N_CATEGORICAL])        |  logit_cat (float64[N_CATEGORICAL])     |
|  5. fuzzy (float64[N_FUZZY * 3])              |  sigma_fuzzy (float64[N_FUZZY * 3])     |
+-----------------------------------------------+-----------------------------------------+
|  APTITUD / FITNESS (float64)                  |  fitness (Menor es mejor)               |
+-----------------------------------------------------------------------------------------+
```

---

### Diagrama 3.3: Ciclo Evolutivo Generacional ($\mu, \lambda$) / ($\mu + \lambda$)

```text
                           +-----------------------------------+
                           | Inicializar Población de Padres  |
                           |   (MU individuos aleatorios)     |
                           +-----------------+-----------------+
                                             |
                                             v
                           +-----------------------------------+
                           |  Evaluar Fitness de los Padres    |
                           +-----------------+-----------------+
                                             |
    +---------------------------------------->----------------------------------------+
    |                                                                                 |
    |                                        v                                        |
    |                      +-----------------------------------+                      |
    |                      |   Generar LAMBDA Hijos (Bucle)    |                      |
    |                      +-----------------+-----------------+                      |
    |                                        |                                        |
    |                                        v                                        |
    |                      +-----------------------------------+                      |
    |                      |  Seleccionar Padre y Clonar Hijo   |                      |
    |                      |     (clone_individual)            |                      |
    |                      +-----------------+-----------------+                      |
    |                                        |                                        |
    |                                        v                                        |
    |                      +-----------------------------------+                      |
    |                      | 1. Adaptar Parámetros Estrategia |                      |
    |                      |    (adapt_scales / adapt_logits)  |                      |
    |                      | 2. Mutar Variables de Decisión    |                      |
    |                      |    (mutate_real, count, etc.)     |                      |
    |                      +-----------------+-----------------+                      |
    |                                        |                                        |
    |                                        v                                        |
    |                      +-----------------------------------+                      |
    |                      |  Evaluar Fitness del Nuevo Hijo   |                      |
    |                      +-----------------+-----------------+                      |
    |                                        |                                        |
    |                                        v                                        |
    |                      +-----------------------------------+                      |
    |                      | ¿Se completaron LAMBDA Hijos?     |                      |
    |                      +-----------------+-----------------+                      |
    |                                       / \                                       |
    |                                 No   /   \  Sí                                  |
    |                                     /     \                                     |
    |                                    +       v                                    |
    |                                            +----------------------------------+ |
    |                                            | Selección de Supervivientes      | |
    |                                            | select_survivors (Rank-Based)    | |
    |                                            |  - Comma: top MU de Hijos         | |
    |                                            |  - Plus: top MU de Padres+Hijos   | |
    |                                            +----------------+-----------------+ |
    |                                                             |                   |
    |                                                             v                   |
    |                                            +----------------------------------+ |
    |                                            | ¿Fin de MAX_GENERATIONS?         | |
    |                                            +----------------+-----------------+ |
    |                                                            / \                  |
    |                                                      No   /   \  Sí             |
    |                                                          /     \                |
    +---------------------------------------------------------+       v               |
                                                               +--------------+       |
                                                               |  Fin del     |       |
                                                               |  Algoritmo   |       |
                                                               +--------------+       |
```

---

## 4. Desglose Componente por Componente

### 4.1. Configuración Centralizada ([`src/es_model/config.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/es_model/config.py))

Define las constantes hiperparamétricas de la simulación:
- **Dinámica de Población:** `MU` = 15 (supervivientes), `LAMBDA` = 100 (descendientes por generación), `SELECTION_MODE` = `"comma"` ($(\mu, \lambda)$) o `"plus"` ($(\mu + \lambda)$).
- **Dimensiones del Genoma:** `N_REAL` (10), `N_COUNT` (5), `N_BINARY` (8), `N_CATEGORICAL` (3), `N_FUZZY` (4).
- **Límites (Bounds):** Rango real $[-5.0, 5.0]$, rango de conteo $[0, 20]$.
- **Constantes de Auto-Adaptación:** `N_TOTAL` (dimensión equivalente total), `SIGMA_FLOOR` = $10^{-6}$ (suelo numérico para evitar congelamiento de paso).
- **Semilla Numérica:** `SEED` = 42 (garantiza reproducibilidad determinista).

---

### 4.2. Kernels Matemáticos Vectorizados ([`src/es_model/kernels.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/es_model/kernels.py))

Contiene las funciones puras sin estado operadas sobre arrays de NumPy:

1. **Tasas de Aprendizaje Global y Local ($\tau', \tau$):**
   $$\tau' = \frac{1}{\sqrt{2 \cdot N_{total}}}, \quad \tau = \frac{1}{\sqrt{2 \cdot \sqrt{N_{total}}}}$$

2. **Adaptación Log-Normal (`adapt_scales`):**  
   Para pasos de mutación ($\sigma_{real}, \lambda_{count}, \sigma_{fuzzy}$):
   $$\sigma_i' = \max\left(\sigma_{floor}, \, \sigma_i \cdot \exp\left(\tau' \cdot \mathcal{N}(0,1) + \tau \cdot \mathcal{N}_i(0,1)\right)\right)$$

3. **Adaptación Logit-Normal (`adapt_logits` y `sigmoid`):**  
   Para probabilidades binarias y categóricas. Ajusta los logits en $\mathbb{R}$ mediante ruido aditivo y aplica la función sigmoide $p = \frac{1}{1 + e^{-logit}}$ para mapear a $(0, 1)$.

4. **Operadores de Mutación por Tipo de Variable:**
   - **Real (`mutate_real`):** Perturbación Gaussiana $x' = \text{clip}(x + \sigma \cdot \mathcal{N}(0,1), \, [L, U])$.
   - **Conteo (`mutate_count`):** Perturbación de Skellam $x' = \text{clip}(x + \text{Poisson}(\lambda) - \text{Poisson}(\lambda), \, [L, U])$.
   - **Binario (`mutate_binary`):** Inversión de bits mediante XOR guiado por Bernoulli($p$).
   - **Categórico (`mutate_categorical`):** Si se activa la probabilidad, se re-muestrea de manera uniforme entre las alternativas restantes $\mathbb{C} \setminus \{x_i\}$.
   - **Difuso (`mutate_fuzzy`):** Perturba los 3 puntos $(a, b, c)$, aplica reflejo especular sobre $[0, 1]$ para evitar desbordamiento y aplica enclavamiento monótono para garantizar $a \le b \le c$ con margen $\epsilon = 10^{-4}$.

---

### 4.3. Motor de Población y Selección ([`src/es_model/engine.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/es_model/engine.py))

- **`Individual`:** Dataclass que reúne todos los bloques del genoma y sus hiperparámetros. Expone la propiedad computada `.genome` para aplanar todos los valores de decisión en un único vector continuo utilizado en logs y gráficas.
- **`create_random_individual`:** Genera un individuo inicial muestreando uniformemente las variables dentro de sus bounds y asignando los valores de estrategia por defecto.
- **`clone_individual`:** Realiza una copia profunda (deep copy) e independiente de las estructuras NumPy para evitar la contaminación entre padres e hijos durante las mutaciones.
- **`mutate_individual`:** Orquesta la secuencia completa de mutación (adaptación de parámetros $\rightarrow$ mutación de variables).
- **`select_survivors`:** Aplica selección determinista basada en ranking. Ordena la población (hijos en modo `"comma"`, padres+hijos en modo `"plus"`) de menor a mayor fitness y selecciona los mejores $\mu$ individuos. La ordenación de Python preserva el orden posicional en empates para máxima reproducibilidad.

---

### 4.4. Punto de Entrada y Coordinación ([`src/es_model/main.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/es_model/main.py))

Proporciona la interfaz de usuario por consola e implementa los 4 flujos principales:
- `model`: Ejecuta la ES Heterogénea completa sobre la función objetivo Sphere.
- `benchmarks`: Ejecuta la batería comparativa frente a *Isotropic ES*, *sep-CMA-ES* y *Standard CMA-ES* sobre la suite de funciones benchmark.
- `smoke`: Realiza una prueba rápida de integración de pocas generaciones y evaluaciones para verificar la integridad del sistema.
- `all`: Ejecuta de manera secuencial los flujos `model` y `benchmarks`.

---

### 4.5. Suite de Benchmarks y Runners ([`src/benchmarks/`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/benchmarks/) y [`src/runners/`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/runners/))

- **`functions.py` ([`src/benchmarks/functions.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/benchmarks/functions.py)):** Implementa 4 funciones objetivo estándar para optimización continua:
  1. *Sphere:* $f(\mathbf{x}) = \sum x_i^2$ (Unimodal, separable).
  2. *Rastrigin:* $f(\mathbf{x}) = 10n + \sum (x_i^2 - 10 \cos(2\pi x_i))$ (Multimodal con alta densidad de óptimos locales).
  3. *Rosenbrock:* $f(\mathbf{x}) = \sum (100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)$ (Valle no separable).
  4. *Ackley:* $f(\mathbf{x}) = -20 \exp(-0.2\sqrt{\frac{1}{n}\sum x_i^2}) - \exp(\frac{1}{n}\sum \cos(2\pi x_i)) + 20 + e$ (Multimodal con cuenca global suave).

- **Runners de Algoritmos ([`src/runners/`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/runners/)):**
  - **`custom_es.py` ([`src/runners/custom_es.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/runners/custom_es.py)):** Adapta el prototipo heterogéneo a problemas continuos de cualquier dimensión $N$, aplicando recombinación intermedia global para parámetros de estrategia y recombinación discreta dual para las variables reales.
  - **`isotropic_es.py` ([`src/runners/isotropic_es.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/runners/isotropic_es.py)):** ES autorregulada clásica con paso escalar isotrópico único.
  - **`sep_cma_es.py` ([`src/runners/sep_cma_es.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/runners/sep_cma_es.py)):** Separable CMA-ES (matriz de covarianza diagonal).
  - **`standard_cma_es.py` ([`src/runners/standard_cma_es.py`](file:///c:/Users/bejmo/Documents/documents/GitHub/TFG_raona/src/runners/standard_cma_es.py)):** CMA-ES estándar con matriz de covarianza completa (actúa como cota superior de rendimiento / upper bound).

---

## 5. Cuadro Resumen de Funciones y Módulos

| Módulo / Archivo | Responsabilidad Principal | Clases / Funciones Clave |
| :--- | :--- | :--- |
| **`config.py`** | Definición de constantes y parámetros. | `MU`, `LAMBDA`, `N_REAL`, `N_COUNT`, `REAL_BOUNDS`, `SIGMA_FLOOR` |
| **`kernels.py`** | Operadores matemáticos vectorizados de mutación. | `adapt_scales()`, `adapt_logits()`, `mutate_real()`, `mutate_count()`, `mutate_fuzzy()` |
| **`engine.py`** | Estructuras de datos y ciclo generacional. | `Individual`, `create_random_individual()`, `mutate_individual()`, `select_survivors()` |
| **`main.py`** | CLI, menús de usuario y bucle principal. | `run()`, `run_model()`, `run_benchmarks()`, `main()` |
| **`functions.py`**| Definición de paisajes de aptitud. | `sphere()`, `rastrigin()`, `rosenbrock()`, `ackley()`, `BENCHMARKS` |
| **`custom_es.py`**| Adaptador de la ES Heterogénea para benchmarks. | `run()` |
| **`isotropic_es.py`**| Baseline de ES isotrópica simple. | `run()`, `IsotropicIndividual` |
| **`sep_cma_es.py`**| Baseline de CMA-ES separable. | `run()` (vía librería `cma`) |
| **`standard_cma_es.py`**| Baseline de CMA-ES completa (Upper Bound). | `run()` (vía librería `cma`) |

---

## 6. Comandos Principales de Uso

### 6.1 Interfaz Interactiva (Recomendado)
```bash
python -m src.es_model.main
```

### 6.2 Modos Directos por Línea de Comandos
```bash
# Ejecutar optimización del modelo con 200 generaciones
python -m src.es_model.main model --generations 200 --log-interval 20

# Ejecutar benchmark en dim 20 con presupuesto de 50,000 evaluaciones
python -m src.es_model.main benchmarks --dim 20 --max-evals 50000 --seeds 5

# Test rápido de humo
python -m src.es_model.main smoke --dim 5 --max-evals 200 --seeds 2
```

---

## 7. Diagrama de Dependencias entre Módulos Python

Este gráfico en texto plano muestra las relaciones de importación directa (`import` / `from ... import ...`) únicamente entre los módulos creados en este repositorio (excluyendo librerías externas de terceros como NumPy o SciPy):

```text
                                +---------------+
                                |   config.py   |
                                +-------+-------+
                                        |
                   +--------------------+--------------------+
                   | (importa)                               | (importa)
                   v                                         v
           +---------------+                         +---------------+
           |  kernels.py   |<------------------------|   engine.py   |
           +-------+-------+        (importa)        +-------+-------+
                   ^                                         ^
                   | (importa)                               | (importa)
                   +--------------------+--------------------+
                                        |
                                +-------+-------+
                                |  custom_es.py |
                                +-------+-------+
                                        ^
                                        | (importa)
   +--------------------+---------------+--------------------+--------------------+
   | (importa)          | (importa)                          | (importa)          | (importa)
   v                    v                                    v                    v
+--+------------+    +--+------------+                    +--+------------+    +--+------------+
| functions.py  |    |  isotropic_es |                    |  sep_cma_es   |    |standard_cma_es|
+--+------------+    +--+------------+                    +--+------------+    +--+------------+
   ^                    ^                                    ^                    ^
   |                    |                                    |                    |
   +--------------------+---------------+--------------------+--------------------+
                                        | (todos son importados por)
                                        v
                                +---------------+
                                |    main.py    |
                                +---------------+
```

### 7.1 Matriz de Dependencias (Qué importa qué)

| Módulo Python (`.py`) | Depende de (Archivos del repo que importa) | Es Usado por (Módulos que lo importan) |
| :--- | :--- | :--- |
| **`src/es_model/config.py`** | *(Ninguno - Módulo Raíz de Configuración)* | `kernels.py`, `engine.py`, `main.py`, `custom_es.py`, `experiment_1.ipynb` |
| **`src/es_model/kernels.py`** | `config.py` | `engine.py`, `custom_es.py`, `experiment_1.ipynb` |
| **`src/es_model/engine.py`** | `config.py`, `kernels.py` | `main.py`, `custom_es.py`, `experiment_1.ipynb` |
| **`src/es_model/main.py`** | `config.py`, `engine.py`, `functions.py`, `custom_es.py`, `isotropic_es.py`, `sep_cma_es.py`, `standard_cma_es.py` | *(Punto de entrada principal CLI / Menú)* |
| **`src/benchmarks/functions.py`** | *(Ninguno - Definición pura de funciones objetivo)* | `main.py`, `experiment_1.ipynb` |
| **`src/runners/custom_es.py`** | `config.py`, `kernels.py`, `engine.py` | `main.py`, `experiment_1.ipynb` |
| **`src/runners/isotropic_es.py`** | *(Ninguno)* | `main.py`, `experiment_1.ipynb` |
| **`src/runners/one_plus_one_es.py`** | *(Ninguno)* | *(Runner independiente de referencia)* |
| **`src/runners/sep_cma_es.py`** | *(Ninguno local)* | `main.py`, `experiment_1.ipynb` |
| **`src/runners/standard_cma_es.py`** | *(Ninguno local)* | `main.py`, `experiment_1.ipynb` |

