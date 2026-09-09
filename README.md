# Formula One Aerodynamic Modeling and Optimization: Wind Tunnel and Sensor Data Analysis

Surrogate modelling and numerical optimization of Formula 1 aerodynamic performance, built from wind tunnel measurements and on-car pressure sensor data.

The work covers five connected problems: fitting a surrogate model for total aerodynamic load from ride heights, optimizing that model under physical constraints, rebuilding the same target from 112 surface pressure sensors, reducing that sensor array to its 20 most informative locations, and minimizing the wall-clock time of a wind tunnel run while still covering the design space.

---

## Problem setup

A scale model is tested in a wind tunnel at a range of front and rear ride heights. Two actuators set the ride heights:

| Variable | Meaning | Constraint | Actuator speed |
|---|---|---|---|
| `H1` | Front ride height | `0.01 ≤ H1 ≤ 0.20` | 0.010 m/s |
| `H2` | Rear ride height | `H1 ≤ H2 ≤ H1 + 0.144` | 0.015 m/s |

A strain gauge balance records the aerodynamic load in the horizontal and vertical directions (`cx`, `cy`), combined into a single target:

$$c_{\text{total}} = \sqrt{c_x^2 + c_y^2}$$

A separate array of 112 pressure taps (`p0 … p111`) records the pressure distribution over the car's surface for the same runs.

---

## The five parts

| Part | Question | Notebook |
|---|---|---|
| **P1** | Build a surrogate model $c_{\text{total}} = f(H_1, H_2)$ | `wind_tunnel_modelling.ipynb` |
| **P2** | Find the $(H_1, H_2)$ that minimizes $c_{\text{total}}$ under the constraints | `wind_tunnel_modelling.ipynb` |
| **P3** | Build a surrogate model $c_{\text{total}} = g(p_0, \dots, p_{111})$ | `sensors_on_car_modelling.ipynb` |
| **P4** | Reduce 112 sensors to the 20 most informative, keeping accuracy | `sensors_on_car_modelling.ipynb` |
| **P5** | Choose N discrete $(H_1, H_2)$ points and a visiting order that minimizes tunnel time while covering the space | `wind_tunnel_modelling.ipynb` |

**Read `wind_tunnel_modelling.ipynb` first.** It performs the outlier analysis that `sensors_on_car_modelling.ipynb` inherits — the two measurements identified there are dropped directly at the top of the second notebook.

---

## Results

### P1 — Ride height surrogate model

Seven candidate models compared under leave-one-out cross-validation (LOOCV was chosen over a train/test split because the cleaned dataset holds only 42 points).

| Model | Params | Training MSE | **Validation MSE** |
|---|---|---|---|
| Linear regression | 3 | 27.014 | 35.274 |
| 2nd-order polynomial | 6 | 7.911 | 18.084 |
| 3rd-order polynomial | 10 | 1.714 | 8.736 |
| **4th-order polynomial** | **15** | **0.443** | **6.110** |
| 5th-order polynomial | 21 | 0.133 | 6.820 |
| 6th-order polynomial | 28 | 0.673 | 11.552 |
| Random forest | — | lower | higher |

The 4th-order polynomial was selected, at a mean validation error of **8.40 %**. The table is a textbook bias–variance curve: the degree-5 fit cuts the training error by a further two-thirds (0.133) while its validation error *rises*, and degree 6 degrades on both. The random forest shows the same pattern more sharply. Validation error is minimized exactly where the model stops gaining and starts memorizing.

### P2 — Constrained optimum

Minimized with `scipy.optimize.minimize` using `trust-constr`, with the four inequality constraints stated above.

| | Value |
|---|---|
| Optimal `H1` | **0.19577** |
| Optimal `H2` | **0.23833** |
| Minimum `c_total` | **6.033** |

The optimum sits against the upper `H1` bound, which is consistent with the raw scatter of `H1` against `c_total`. Sweeping the initial guess across all 42 training points showed convergence to this same minimum from nearly every start, with one uncommon local minimum.

Interpreted in racing terms: minimizing the *combined* load favours a setup for circuits dominated by long straights. A high-downforce circuit would need a floor constraint on $c_y$ rather than a pure minimization.

### P3 — Pressure sensor surrogate model

A high-dimensional, low-sample-size problem: **112 features, 42 samples**. Ridge regression (α = 0.05) was selected.

The accuracy was high enough to be suspicious, so a substantial part of this section is spent trying to break the result rather than accept it — LOOCV, a shuffled hold-out test set excluded from the cross-validation entirely, and a physical sanity check. The sanity check treats the area under each parallel-coordinates line as a proxy for integrating pressure over the car's surface; the run with the largest such area (index 7) is also the run with the highest predicted $c_{\text{total}}$ (index 7).

### P4 — Sensor reduction, 112 → 20

Two selection strategies compared against two random baselines, scoring the 20-sensor model's predictions against the 112-sensor model's predictions on the same test set.

| Method | MSE vs 112-sensor model | Total absolute error |
|---|---|---|
| **Recursive Feature Elimination** | **0.0158** | **4.600** |
| Feature Importance Ranking (Ridge coefficients) | 0.1294 | 12.079 |
| Random 20 sensors | worse | worse |
| Random 20 *consecutive* sensors | far worse | far worse |

RFE wins. The two random baselines are informative in their own right: scattered random sensors do reasonably well, while 20 *consecutive* sensors do badly — coverage across the car matters more than sensor count.

Selected sensors:

```
p12  p13  p14  p23  p51  p53  p54  p55  p63  p64
p65  p69  p71  p85  p91  p92  p93  p94  p95  p107
```

Overlaying both methods' selections on the parallel-coordinates plot shows 14 shared sensors, with RFE weighting the forward surface more heavily and picking up one sensor from the rear region that FIR ignores.

### P5 — Minimizing wind tunnel time

N = 30 points generated across the constrained $(H_1, H_2)$ space, then five algorithms used to order the visits. Move time between two setpoints is set by the slower actuator: $t = \max(\Delta H_1 / v_1,\ \Delta H_2 / v_2)$.

| Algorithm | Total run time |
|---|---|
| **Greedy** | **54.14 s** |
| Simulated annealing | 68.03 s |
| Local search | 82.43 s |
| Random search | 84.68 s |
| Naive (random ordering) | 151.87 s |

Against the 89.64 s recorded in the original tunnel run, the greedy schedule of 30 points is a **39.6 % reduction**. Running greedy on the original 42-point space alone gives 83.63 s, so roughly half the saving comes from the ordering and half from the reduced, better-distributed point set.

Greedy beating simulated annealing is the interesting result here, and it is not an accident: over a roughly uniform space, a distant setpoint can only be reached by physically travelling past nearer ones, so the locally optimal choice tends to be globally sound. Random search and local search both stall in local minima with no exploration mechanism to escape.

A method for *quantifying* space coverage was designed and then deliberately discarded — refitting a model on points labelled by the P1 model leaks the original distribution into the evaluation. Coverage is therefore argued visually, and the limitation is stated rather than papered over.

---

## Repository contents

```
wind_tunnel_modelling.ipynb        P1, P2, P5 — data cleaning, ride height surrogate,
                                   constrained optimization, run scheduling

sensors_on_car_modelling.ipynb     P3, P4 — pressure sensor surrogate, sensor reduction
```

## A note on the data

The source workbook (`ML_test.xlsx`) is **not included** in this repository. These notebooks are published as a written analysis with all outputs, figures, and results preserved inline — they are meant to be read, not re-executed.

## A note on the figures

The 3-D surface and scatter plots are built with Plotly and **will not render in GitHub's notebook preview**, which strips the JavaScript. They appear as blank space. To see them, open the notebooks in Jupyter or [nbviewer](https://nbviewer.org/). All Matplotlib and Seaborn figures render normally.

## Stack

`pandas` · `numpy` · `scikit-learn` · `scipy` · `tensorflow` / `keras` · `matplotlib` · `seaborn` · `plotly` · `openpyxl`

## Methods used

Z-score outlier detection · leave-one-out cross-validation · polynomial regression · ridge / lasso regression · random forest regression · feed-forward neural network · constrained numerical optimization (`trust-constr`) · recursive feature elimination · coefficient-based feature importance · greedy / local search / random search / simulated annealing
