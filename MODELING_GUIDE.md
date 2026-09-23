# Modeling Guide: Predicting Plasma Shape

## What Are We Even Doing?

Imagine you're baking a cake in an oven you can't open. You can't see the cake. You can't touch it. But you **can** read the oven's control panel -- the temperature dial, the fan speed, the timer. Your job: **predict exactly what the cake looks like inside**, just from reading those dials.

That's basically our problem, except:
- The "cake" is a blob of super-hot plasma (100 million degrees)
- The "oven" is a tokamak (a donut-shaped magnetic bottle)
- The "dials" are the currents flowing through electromagnets
- The "shape of the cake" is a 65x65 pixel image called a **flux map**

---

## The Target: What Exactly Is a Flux Map?

### The 65x65 Grid

Our target is called `efit_psirz` -- a 65x65 grid of numbers. Each number represents the **magnetic flux** at that point in space. Think of it like a topographical map:

- **High values** (red in our plots) = strong magnetic field pushing outward
- **Low values** (blue in our plots) = the magnetic "valley" where plasma lives
- **Contour lines** = if you drew lines connecting equal values, like elevation lines on a hiking map, you'd see the shape of the magnetic cage

The plasma lives inside the deepest "valley" of this map. The shape of that valley IS the shape of the plasma.

### One Image Per Timestep

A single fusion experiment ("shot") lasts a few seconds, but a LOT happens. We get roughly 300 of these 65x65 snapshots per shot, each at a different moment in time. So one shot gives us ~300 target images.

### Beyond the Flux Map: Scalar Targets

The flux map is the *primary* target, but the corrected dataset also ships a handful of
**EFIT scalar labels** — one number per timestep — that summarise the equilibrium. You *submit*
only `q95` and `betaN`; the others are training supervision (at scoring time the axis, shape and
`li` are derived from your submitted flux map):

| Column | Plain English |
|--------|---------------|
| `efit_beta_n` | Normalised beta β_N — roughly, how much plasma pressure you're holding for the field you're using (an efficiency/stability number) |
| `efit_li` | Internal inductance ℓi — how peaked vs. broad the current profile is |
| `efit_q95` | Safety factor at the 95% flux surface — a key stability number (how many times field lines wind the long way per short-way loop near the edge) |
| `efit_r_axis`, `efit_z_axis` | Where the magnetic axis (the very center of the "valley") sits, in meters |
| `magnetics_dsep` | Context only, not scored. DIII-D: EFIT's a-file separatrix↔limiter clearance. MAST: a *different* quantity — divertor balance. EFIT-derived, so withheld on test despite its `magnetics_` name |

These are present in `diii_d_train` and **withheld on the test splits**, exactly like
`efit_psirz`. `experiments.py` predicts them with a simple per-scalar Ridge baseline and
reports R² for each (see `results/scalar_r2.png`). Two practical notes:

- **They live on a totally different scale from the flux pixels.** Predict them as their
  own regression targets (raw values), not by squishing them into the flux-map pipeline.
- **`dsep` is no longer scored, and the two machines' columns are not the same quantity.**
  DIII-D's is a separatrix↔limiter clearance whose sign encodes the configuration; MAST's is
  divertor *balance* (δR_sep), which straddles zero on ordinary diverted plasmas — **never read
  its sign as a diverted/limited flag**. It stays in `train` as context; MAST's still carries
  `NaN` and a `-1.0` sentinel, so mask before using it for anything.
- **The five scored scalars can also be `NaN`** on frames with incomplete reconstruction
  (a few startup/rampdown frames, plus some frames on a handful of DIII-D shots whose
  scalars sat on a slightly offset EFIT time base). All five share the same valid frames,
  so one `NaN` mask covers them. The starter code masks per-scalar automatically.

A natural extension of the neural-net baselines is a small **scalar head** for the two
*submitted* scalars (`q95`, `betaN`), plus — optionally — auxiliary heads trained on the other
labels purely as extra supervision for the flux decoder. Remember: only `q95`/`betaN`
predictions are scored as values; every geometric scalar is derived from your submitted flux
map at scoring time, so auxiliary heads help only insofar as they make your ψ better.

### Why the Plots Might Look Different from the dFL Labeler

If you've used the dFL (Data Fusion Labeler) to visualize this data, you may notice model output plots can look different. Here's why:

**In the dFL Labeler (`fusion_data_provider.py`):**
- Uses **contour lines** (like a topographical map with labeled elevation lines)
- X-axis is **Major Radius R (meters)** -- the physical distance from the center of the tokamak
- Y-axis is **Vertical Position Z (meters)** -- how far up or down
- These are real-world physical coordinates
- Uses Plotly for interactive visualization

**In a typical model training script:**
- Might use a **heatmap** (colored pixels, no contour lines)
- X-axis might be **pixel index 0-64** -- just the column number in the grid
- Y-axis might be **pixel index 0-64** -- just the row number
- These are array indices, not physical coordinates
- Often uses Matplotlib for static images

**They show the exact same data.** It's like looking at the same mountain on a topographical map (contour lines with labeled elevations) vs. a satellite thermal image (colored pixels). Same mountain, different visualization style.

The dFL version is more useful for physicists (real coordinates, contour lines). A heatmap version is more useful for machine learning (raw pixel grid, easier to compare predicted vs. actual).

---

## The Inputs: What Information Can Your Model Use?

### Magnetics (Coil Currents)

For DIII-D, there are **21 signals** measuring the current flowing through each electromagnet:

- **ECOILA**: The main solenoid -- a big coil running through the center of the donut. It drives the plasma current. Think of it as the "engine."
- **bcoil**: The toroidal field coil -- wraps around the donut to keep the plasma stable. Think of it as the "guardrails."
- **F1A through F9B** (18 coils): Shaping coils arranged around the outside of the tokamak. These are the "sculptor's hands" -- they push and pull the plasma into the right shape. A-coils are on top, B-coils are on the bottom.
- **plasma_current**: How much electrical current the plasma itself is carrying (yes, the plasma conducts electricity!).

For MAST, there are **14 signals** with a different coil arrangement (P2L-P6U, Solenoid, TF, EFPS, plasma_current). See the README for the full list.

### Thomson Scattering (Laser Diagnostics)

Both machines also have laser diagnostics that measure the temperature and density of the electrons inside the plasma. These could provide additional information about the plasma state that isn't captured by the magnet currents alone. See the README for details on these columns.

### The Core Challenge

```
21 numbers (magnet currents) --> 4,225 numbers (65x65 pixel image)
```

That's a LOT of outputs for very few inputs. This is what makes the problem interesting.

---

## Challenge 1: Different Time Bases

The magnet measurements and the flux map snapshots are recorded at **different sampling rates**.

- The magnets are measured ~50,000 to ~480,000 times per shot (very fast sampling — DIII-D has
  two populations, ~70% of shots at 20 kHz and the rest at 2 kHz, so read `magnetics_time` per
  shot rather than assuming a length)
- The flux maps are computed only ~300 times per shot (expensive calculation)
- Thomson scattering has its own separate time base (~1,800 core samples, ~200 edge samples)

It's like having a security camera that records at 30fps and a thermometer that only logs every 10 seconds. To use them together, you need to decide how to align them.

### Possible Approaches

**Interpolate inputs to target times.** For each of the ~300 flux map timestamps, estimate what each magnet current was at that exact moment. This is the simplest approach -- draw a smooth line through the magnet data points and read off the value at the time you need. This way, the target (flux maps) is never modified -- only the inputs are resampled.

> **⚠️ One data erratum matters exactly here:** on ~69% of DIII-D shots the shipped
> `magnetics_plasma_current_times` axis is wrong (the Ip *values* are fine), so interpolating Ip
> onto `efit_times` with it returns pre-plasma noise instead of ~1 MA. Use
> `data_fixes.fix_d3d_ip_times(row)` to get the corrected per-shot axis — see the README's
> *Data errata* section. The baseline in `experiments.py` already applies it.

**Important: you should NOT interpolate the targets.** The flux maps are the ground truth. Modifying them (e.g., resampling to a different time grid) would introduce artifacts into the labels your model is learning from.

**Use the raw time series as input.** Instead of collapsing each magnet to a single value per timestep, you could feed a model a window of the raw high-frequency magnet data (e.g., the last 100ms of all 21 signals). This gives the model more information but requires sequence-aware architectures (RNNs, Transformers, 1D CNNs).

**Downsample inputs to a fixed grid.** Resample all magnetics to a common, lower-frequency time grid (e.g., every 1ms), then look up the values closest to each EFIT time. Cruder than interpolation but simpler.

---

## Challenge 2: High-Dimensional Output

Predicting 4,225 numbers per sample is a lot. Here are some strategies:

### PCA Compression (Recommended Starting Point)

**PCA (Principal Component Analysis)** finds the most important "building blocks" of the flux maps. Think of it like faces: every human face has eyes, a nose, a mouth arranged in roughly the same way. You don't need to describe every pixel -- you can say "wide eyes, narrow nose, big smile" and reconstruct most of the image.

PCA does the same thing with flux maps:
- **Component 1**: The most common pattern (e.g., "overall flux level"). This alone explains ~92% of the variation across all flux maps.
- **Component 2**: The next most important pattern. Adds another ~6%.
- With just **2-3 components**, you capture ~99% of the variation.

This means instead of predicting 4,225 numbers, a model only needs to predict **20-50 PCA coefficients**, then PCA reconstructs the full image. This dramatically reduces the problem size and is particularly effective with traditional ML models (linear regression, random forests, etc.).

### Direct Image Prediction (Neural Networks)

Neural networks (CNNs, U-Nets, etc.) can predict the full 65x65 grid directly without PCA. This avoids any information loss from compression, but requires much more training data to work well. See the note on model capacity below.

### Autoencoder Latent Space

Train an autoencoder to compress flux maps into a small latent vector, then train a separate model to predict that latent vector from inputs. Similar to PCA but can capture nonlinear structure.

---

## Possible Model Approaches

Here are some approaches ranked roughly from simplest to most complex. We encourage you to start simple and build up -- a good linear model often beats a poorly-tuned neural network.

### Linear Regression (Start Here)

The "Hello World" of machine learning. It says:

> Each output is a weighted sum of the inputs, plus a constant.

For each PCA coefficient, the model learns something like:

```
PCA_coefficient_1 = 0.3 * ECOILA + (-0.1) * bcoil + 0.05 * F1A + ... + 2.1
```

Multiply each input by a learned weight, add them up, add a bias. No hidden layers, no activation functions, no magic.

**Why does this work at all?** The magnetic field is (approximately) a linear function of the currents producing it. Turn up a coil by 10%, and the field it produces goes up by roughly 10%. So a linear model is actually a decent first approximation for this physics problem.

In our demo experiments (run in `--quick` mode with only 3 shots / ~264 training samples), linear regression trains in **<0.1 seconds** and reaches **SSIM ≈ 0.4–0.85** depending on which 3 shots are drawn — the variance is huge at this sample size, so treat `--quick` numbers as a smoke test, not a benchmark. With more shots the scores stabilize and climb (Ridge/RandomForest/MLP reach SSIM ≈ 0.87–0.90 on ~100 shots).

### Ridge Regression

Same as linear regression but with a **penalty for large weights**. The model has to balance two goals:

1. **Fit the data well** (make accurate predictions)
2. **Keep the weights small** (don't rely too heavily on any single input)

The tradeoff is controlled by a parameter called **alpha**. You can use `RidgeCV` (from scikit-learn) to automatically try many alpha values and pick the best one via cross-validation -- like trying 10 different spice levels and picking the one your taste-testers like most.

This consistently beats plain linear regression by preventing overfitting.

### Tree-Based Models (Random Forest, Gradient Boosting)

Tree-based models can capture **nonlinear** relationships that linear models miss. For example, maybe the effect of ECOILA depends on what bcoil is doing -- a tree can learn "if ECOILA > 5000 AND bcoil < 2000, then..." while a linear model cannot.

Use with PCA targets. These models can be slow when predicting many outputs -- `RandomForestRegressor` from scikit-learn supports multi-output natively, but you may also want to try `HistGradientBoostingRegressor` with a `MultiOutputRegressor` wrapper.

### Neural Networks (MLPs)

A multi-layer perceptron stacks several layers of "linear regression + nonlinear activation." This allows the model to learn complex, nonlinear mappings.

**A critical lesson from our experiments:** When using neural networks, the choice of output space matters enormously.

In our demo experiments (3 shots, ~264 training samples -- `--quick` mode):

| Approach | Parameters | MSE | Notes |
|----------|-----------|-----|-------|
| sklearn MLP predicting 20 PCA coefficients | 41,000 | 0.005 | Works with small data |
| PyTorch MLP predicting 4,225 raw pixels | 5,000,000 | 0.239 | Needs much more data |

Same architecture concept, but the PCA version has **122x fewer parameters** and achieved **50x lower error** in this small-data demo. The raw-pixel version overfits with so few samples. With the full dataset (11,000+ shots), neural networks predicting raw pixels would have enough data to train properly and could surpass PCA-based approaches. But if you're starting out or working with limited data, **PCA targets are a great practical choice even with neural networks.**

### Convolutional Decoders

An FC (fully-connected) layer maps your input features to a small spatial tensor (e.g., 32 channels at 4x4 resolution), then a series of `ConvTranspose2d` layers upsample it to 65x65. This introduces **spatial structure** into the network -- the model learns that neighboring pixels are related, rather than treating each pixel independently.

### Sequence Models (RNNs, Transformers)

Instead of predicting one flux map from one set of magnet currents, you could model the **entire time series**. Feed the model a sequence of magnet states and predict a sequence of flux maps. This lets the model learn temporal dynamics -- how the plasma evolves over time.

### The Cross-Machine Challenge: Learning the Physics, Not the Machine

The dataset includes both DIII-D and MAST shots. These have completely different coil configurations, different flux geometries, and different diagnostic setups. The deepest challenge here is not transfer learning between machines — it's this:

> **Can you learn the underlying physics of equilibrium reconstruction well enough that your model works on ANY tokamak, even one it has never seen?**

Traditional EFIT reconstruction solves the **Grad-Shafranov equation**, which determines the equilibrium from just a few physics ingredients:
1. **Pressure profile** p($\psi$) — how hot and dense is the plasma at each flux surface?
2. **Current profile** FF'($\psi$) — how is the current distributed?
3. **Boundary conditions** — what are the external fields doing?

If you could compute good proxies for these from any machine's diagnostics, the same model should work everywhere.

### Synthetic Diagnostics: Machine-Agnostic Inputs

Instead of feeding raw coil currents (which are machine-specific), consider deriving **physics quantities** that any tokamak can provide:

**From Thomson scattering (both machines have this):**
- Electron pressure profile: $p_e = n_e \times T_e$ at each spatial channel
- Peak pressure and its location
- Pressure gradient (related to bootstrap current)
- Profile shape parameters (peakedness, width, pedestal height)
- Total stored energy proxy: $\sum n_e \times T_e$ integrated over channels

**From magnetics (universal concepts):**
- Plasma current $I_p$ (both machines measure this directly)
- Safety factor proxy $q \sim B_t / B_p$ (from toroidal field and plasma current)
- Total external poloidal field energy (aggregate of shaping coil currents)
- Up-down asymmetry of shaping fields

The idea is that these derived quantities are **closer to the physics** than raw coil currents. A model trained on pressure profiles, current indicators, and field energy should generalize better because it's learning the Grad-Shafranov physics, not the wiring diagram of a specific machine.

### Why This Is Hard (and Interesting)

In our demo (`model_experiments/cross_machine.py`), a simple coil-mapping approach trained on DIII-D achieved SSIM=0.83 on DIII-D but only SSIM=0.10 on MAST — essentially failing to transfer. This is because:

- The **flux value ranges** are very different (DIII-D: ~-0.25, MAST: ~+0.05)
- The **flux geometry** is fundamentally different (conventional vs spherical tokamak)
- The **PCA basis** learned from one machine can't represent the other's structure
- Raw coil aggregates don't capture the physics, just the engineering

**Ideas to explore:**
- Derive physics-based features (pressure profiles, q estimates) and train on those instead of raw coils
- Normalize flux maps per-sample (e.g., subtract mean, divide by range) so the model predicts *shape* rather than absolute values
- Use separate PCA bases per machine but a shared physics feature space
- Train on both machines simultaneously with synthetic diagnostics as input
- Explore whether Thomson scattering data alone (without any magnetics) can determine the equilibrium shape

---

## Tips and Pitfalls

### Split by Shot, Not by Timestep

Timesteps within a single shot are highly correlated -- the plasma doesn't change much in 3ms. If you randomly split timesteps across train/test, the model can "cheat" by memorizing patterns from the same shot. **Always split at the shot level** to get honest evaluation.

### Don't Interpolate the Targets

The EFIT flux maps are your ground truth. Interpolating, resampling, or smoothing them introduces artifacts. Keep the targets exactly as they are. Only modify the inputs (magnetics, Thomson) to align with the target timestamps.

### Normalize Your Inputs

Magnetics signals have very different scales (DIII-D plasma current peaks near ~1,000 kA while the shaping coils run ~140 kA·turn, and MAST's P3/P6 coils sit near ~3 kA·turn). Use `StandardScaler` or similar normalization so the model doesn't weight large-magnitude signals more just because they have bigger numbers.

### Start Small, Scale Up

With 11,449 DIII-D shots and 3,116 MAST shots available, you have plenty of data. But loading and processing all of it takes time. Start with 10-20 shots to get your pipeline working, then scale up.

### Watch for NaN Values

Some signals may have NaN or Inf values at certain timesteps. Check for and handle these before training -- a single NaN can corrupt an entire training batch.

---

## Evaluation Metrics

**The official leaderboard score is a composite** (flux map, the two predicted
scalars, boundary, and flux-map consistency):

```
S_model = 0.55 · R²_ψ  +  0.15 · R²_{q95,βN}  +  0.10 · (1 − D_LCFS)  +  0.20 · Consistency
```

| Term | What it measures | Good |
|------|------------------|------|
| **R²_ψ** | Global R² of the flux map over all (R,Z) points × timesteps × shots (clipped ≥ 0) | → 1 |
| **R²_{q95,βN}** | Mean pooled R² of the two *submitted* scalars, `q95` and `betaN` (clipped ≥ 0) | → 1 |
| **D_LCFS** | Symmetric Hausdorff distance between the LCFS contours extracted from your ψ and the true ψ, normalized by mean true LCFS major radius (clipped ≤ 1). You don't submit a contour — predict a good ψ. | → 0 |
| **Consistency** | Mean agreement of the seven ψ-derived scalars (`R_axis, Z_axis, κ, δ_top, δ_bot, V, li`): each is computed from *your* ψ and scored against the same computation on the *true* ψ (pooled R², clipped ≥ 0, averaged). You don't submit any of them — your flux map has to imply them. | → 1 |

Cross-machine transfer (Award #2) is `G_ratio = S_model(MAST) / S_model(DIII-D)` among entries with
a DIII-D composite `S_model ≥ 0.85` (raised 2026-08-18 from `R²_ψ > 0.6`, see issue #7; the scorer
enforces the gate — below it `G_ratio` is reported as 0). DIII-D and MAST are scored separately. **R² can be negative** before
clipping — that means the model is worse than predicting the mean, and it scores 0 in the composite.

**Practical consequence of the Consistency term:** a scalar regression head can no longer earn
points on the ψ-derived quantities — only `q95` and `βN` are predicted as values (they need
`F(ψ)`/`p(ψ)`, which the flux map cannot contain). The training labels (`efit_li`,
`efit_r_axis`, …, `magnetics_dsep`) remain useful as *auxiliary supervision* to shape your ψ
decoder, but at scoring time everything geometric is read off your submitted flux map. `dsep` is
**not scored at all** — it is context only, and the two machines' columns are different physical
quantities. The corpus is diverted-only, so every scored frame has a well-posed derivation; a
perfect ψ scores `Consistency = 1` by construction.

**Diagnostic metrics (not the score).** The baselines in `experiments.py` also print **MSE / MAE /
SSIM** on the flux map. These are quick intuition proxies — SSIM in particular tells you whether the
overall *shape* is right rather than per-pixel values — but the leaderboard ranks on `S_model`, so
treat them as sanity checks, not the objective.

---

## Glossary

| Term | Plain English |
|------|---------------|
| **Flux map (psirz)** | A 65x65 pixel "photo" of the magnetic field shape at one moment in time |
| **EFIT** | The algorithm that computes flux maps from sensor data (our ground truth) |
| **Tokamak** | A donut-shaped machine that uses magnets to hold hot plasma |
| **Interpolation** | Estimating a value between known data points (connecting the dots smoothly) |
| **PCA** | A compression technique that finds the most important patterns in data |
| **R2 score** | How much of the variation your model explains (1.0 = perfect, 0 = useless, negative = worse than guessing the average) |
| **SSIM** | Structural Similarity -- how similar two images look to a human (1.0 = identical) |
| **MSE** | Mean Squared Error -- average of (prediction - actual)^2 (lower = better) |
| **Ridge Regression** | Linear regression with a penalty for large weights (reduces overfitting) |
| **Cross-Validation (CV)** | Automatically testing multiple settings and picking the best one |
| **Overfitting** | When a model memorizes training data instead of learning general patterns |
| **Regularization** | Any technique that prevents overfitting (Ridge's weight penalty is one example) |
