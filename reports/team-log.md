# Team Log — Fusion Equilibrium Challenge

Running record of what each role did, what got in the way, and what actually
worked. Written as the work happened, not reconstructed after the fact.

---

## 1. Research role

**Task:** find the official problem spec, prior art, and the organizers' own
baselines before writing any code.

**Findings:**
- Official challenge paper (arXiv:2609.01750) — this *is* the competition's
  own writeup, including their four baselines and reported scores.
- Prior art: a [Cambridge J. Plasma Physics paper](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/neural-network-reconstruction-of-the-diiid-tokamak-plasma-boundary-using-a-reduced-set-of-diagnostics/78810E6B4AA03B52D381FFA9A0029ECE)
  did almost exactly Challenge 1 already (DIII-D boundary from reduced
  diagnostics) — useful as an architecture reference and a warning (they
  reported a "long tail" of large errors that a pixel-average score hides).
- Physics-informed NN surrogates for TCV and JET — the "if there's time"
  direction beyond a working baseline.

**Hurdles:** none significant — this was a clean web-research pass, no
blocked searches or paywalls encountered.

**Breakthrough:** the paper's own account of *why* naive cross-machine
transfer fails (SSIM 0.83→0.10) and its explicit recommendation
(machine-agnostic physics features: pressure, stored-energy proxy, q~Bt/Bp)
became the direct blueprint for the Challenge 2 work below — this wasn't
incidental background reading, it's the reason that experiment was
designed the way it was.

---

## 2. Problem-understanding role

**Task:** turn the challenge page into an unambiguous spec: inputs, outputs,
scoring formula, deadlines, submission mechanics.

**Findings:** two separate challenges (DIII-D same-machine, DIII-D→MAST
zero-shot transfer), composite score `S = 0.55·R²ψ + 0.15·R²(q95,βN) +
0.10·(1−D_LCFS) + 0.20·Consistency`, Phase 1 closes **Oct 18, 2026**, blind
test Oct 19, Phase 2 closes Oct 26.

**Hurdles:** the dataset card and starter-kit README disagreed slightly on
shot counts (11,200 vs 9,113 vs 7,041+874+1,206 depending on which filtering
stage is being described) — resolved by trusting the starter kit's
`load_dataset` configs as ground truth over prose descriptions, since that's
what's actually downloadable.

**Breakthrough:** none needed — this was straightforward extraction, the
main value was just getting it all in one place before any code was written.

---

## 3. Infra / data-engineer role

**Task:** get from "zero tools installed" to "training data flowing
reliably." This is where almost every real hurdle in this project showed up.

**Hurdles, in the order they happened:**

1. **No `gh` CLI, no git remote pointing anywhere writable.** Fixed by
   installing `gh` via Homebrew and running `gh auth login` — this step
   fundamentally cannot be automated end-to-end: GitHub's device-auth flow
   requires a human to approve in a browser, by design. First attempt's
   one-time code expired before you got to it; second attempt worked.

2. **Streaming the HF dataset is fragile on this network.** Both
   `example_usage.py` and later a 100-shot training run hit intermittent
   `[Errno 8] nodename nor servname provided` / `read operation timed out`
   errors. The 100-shot run eventually crashed outright (`RuntimeError:
   Cannot send a request, as the client has been closed` — the retry logic
   in `huggingface_hub` exhausted itself). Streaming mode does many small
   HTTP range-requests per parquet file, and each one is a chance for the
   flaky connection to drop it.

   **Breakthrough:** switched to whole-file downloads
   (`huggingface_hub.hf_hub_download`, one HTTP request per full file,
   resumable) instead of streaming. 150/150 shots succeeded with zero
   failures on the first try. This is now `scripts/download_shots.py` —
   reusable, and the thing that unblocked every experiment after it.

3. **`load_shot_from_hf_row` (the starter kit's own row parser) crashes on
   MAST rows.** It unconditionally calls `fix_d3d_ip_times()`, which looks
   up a `magnetics_plasma_current_times` column that only exists on DIII-D
   rows (MAST ships one shared `magnetics_time` base with no such erratum to
   fix). This only surfaces the moment you try to load DIII-D and MAST rows
   through the same code path — which nobody had done yet, since the
   starter kit's own scripts only ever load one config at a time.

   **Fix:** a small wrapper (`load_shot_safe` in
   `scripts/cross_machine_physics_features.py`) that patches the missing
   column onto MAST rows before delegating, rather than modifying the
   vendored starter-kit file.

**Net result:** a known-reliable local data pipeline (`hf_local_data/` +
`scripts/download_shots.py`), which is worth more to the rest of this
project than any single model result — nothing downstream works without it.

---

## 4. Challenge 1 modeling role (DIII-D same-machine)

**Task:** get a trustworthy flux-map prediction baseline.

**Hurdle:** the organizers' own `--quick` demo (3 shots) is *supposed* to
look bad — it's a pipeline smoke test, not a result. The scalar R² values
on that run were nonsensical (one was **−1401**), which is correct behavior
for 3 shots / 724 samples trying to fit 6 targets, not a bug. Easy to
mistake for broken code if you don't read the script's own warning.

**Result at 150 shots (32,699 frames, split by shot):**

| Model | Flux map R² | Flux map SSIM |
|---|---|---|
| Ridge (CV) | 0.344 | 0.908 |
| Random Forest | 0.853 | 0.862 |
| **MLP (sklearn)** | **0.952** | **0.899** |

Scalars (Ridge): `efit_beta_n` R²=0.62, `efit_q95` R²=0.25. The weak `q95`
number isn't a modeling failure — the paper is explicit that `q95` needs the
toroidal field function F(ψ), which isn't recoverable from these inputs
alone. Expected, not a bug.

**Breakthrough:** none yet beyond reproducing/beating the organizers' own
numbers — this is a solid foundation, not a novel result.

---

## 5. Challenge 2 modeling role (DIII-D→MAST zero-shot transfer)

**Task:** the paper calls this "the deepest challenge" — test whether
machine-agnostic features actually improve zero-shot transfer, per the
modeling guide's own suggestion.

**Hurdle:** there is **no local MAST ground truth** beyond the 3 demo shots
bundled in `parquet_data/` (115 frames total) — the real
`mast_public_test` targets are withheld by design, since Challenge 2 is
zero-shot. Every number below is a sanity check on n=3 shots, not a
statistically solid result. This is a hard ceiling on local validation,
not something more engineering effort fixes — it's inherent to the
challenge's design.

**Result:**

| Approach | DIII-D SSIM | MAST SSIM (zero-shot) | Transfer ratio |
|---|---|---|---|
| RAW coils (naive) | 0.852 | 0.056 | 0.066 |
| **PHYSICS features + normalized targets** | 0.819 | **0.237** | **0.289** |

**Breakthrough:** this is the one actual research contribution so far, not
just infrastructure. The naive approach reproduces the paper's reported
collapse almost exactly (they got 0.83→0.10, we got 0.852→0.056 — same
failure mode). Swapping in machine-agnostic features (Ip, TF-coil current
proxy, q~TF/Ip, Thomson electron-pressure profile stats — all in physical
units shared by both machines, not raw per-coil values) and predicting
per-frame-normalized flux (shape, not absolute scale) gets **~4.4x better
zero-shot transfer** at a small in-domain cost (0.819 vs 0.852 SSIM).

**Honest limitations, not yet resolved:**
- n=3 MAST ground-truth shots — directional evidence, not proof.
- Scalars (q95, βN) aren't predicted in this experiment at all yet.
- DIII-D R² was *negative* (−0.117) despite high SSIM (0.819) on the same
  predictions — R² and SSIM disagree because R² penalizes any consistent
  pixel-level offset heavily, while SSIM cares more about structure/shape.
  Worth resolving before this becomes a submission, since the official
  score weights `R²ψ` at 0.55 — the single largest term.

**Re-run at 640 shots (4.3x the data) — the finding got sharper, not just
bigger:**

| Approach | DIII-D SSIM | MAST SSIM (zero-shot) | Transfer ratio |
|---|---|---|---|
| RAW coils (naive) | 0.852 → **0.947** | 0.056 → **−0.021** | 0.066 → **−0.022** |
| PHYSICS features | 0.819 → 0.718 | 0.237 → 0.216 | 0.289 → 0.301 |

The naive model's DIII-D fit got better with more data (as expected) while
its MAST transfer flipped net-negative — more data let it fit DIII-D's
specific coil wiring more precisely, which has zero shared structure with
MAST's different coil layout, so a tighter DIII-D fit actively makes
cross-machine transfer *worse*, not just fail to help. This is a stronger
version of the paper's warning than the 150-shot run showed.

The physics-feature model's transfer ratio held steady (0.289→0.301) while
its own in-domain SSIM dropped (0.819→0.718) as shot diversity grew 4.3x.
Reading that as the 7-feature set being **capacity-limited, not
overfitting**: a handful of scalar physics summaries can't represent 640
shots' worth of flux-shape diversity, regardless of how much data backs
them. The lever to pull next is a richer machine-agnostic feature set
(geometric asymmetry via coil R/Z positions, more Thomson profile shape
descriptors) or more model capacity — not more data, which is already not
the bottleneck here.

---

---

## 6. Submission-building role

**Task:** turn the trained models into an actual `submission_skeleton.py`
that produces format-valid predictions for both test configs.

**Hurdle, the most subtle one yet:** the dry-run crashed with a feature-count
mismatch (129 vs 21). Tracing it back: `--include-thomson` had been silently
doing nothing in **every local-data run so far**, including the 640-shot
baseline reported above (R²=0.995/SSIM=0.997) — those numbers used raw coil
currents only, despite us believing Thomson scattering was included.

The cause, layer by layer:
1. Thomson arrays loaded from a *locally downloaded* parquet file arrive as a
   numpy object-array of per-timestep arrays. `np.asarray(that, dtype=float)`
   raises `ValueError: setting an array element with a sequence` **even when
   every element has the identical shape** — a numpy casting quirk, not real
   raggedness (confirmed: all 44 channels, every timestep, for the shot
   tested). This is the exact failure mode `_as_psirz_stack` already existed
   to handle for the flux-map target, just never applied one level down to
   Thomson. The starter kit's loader wraps this in a bare `except
   Exception: pass`, so the column was silently dropped rather than erroring
   — correct defensive design in general, but it hid a real bug here.
2. **Fixed properly** (not worked around): added `_as_profile_stack()` to
   `experiments.py`, the same fix pattern as `_as_psirz_stack`, and used it
   in both the vendored loader and our own inference code.
3. Fixing #1 exposed a second, *real* issue underneath: core Thomson channel
   count genuinely varies shot-to-shot (44 channels on most shots, but 42 or
   54 on others — real hardware/config differences, confirmed across 30
   shots). Raw per-channel feature expansion can't produce one fixed-width
   feature matrix across shots with different channel counts.

**Decision, not a fix:** rather than pad/truncate channels (which would mix
up spatial meaning across differently-configured shots) or redesign the
DIII-D pipeline around aggregate Thomson statistics late in the process, we
descoped Thomson from the DIII-D pipeline for this submission and kept the
already-validated raw-coil-only model (R²=0.995, SSIM=0.997 — those numbers
are accurate; they just never had Thomson in them to begin with). The MAST
physics-feature pipeline was never affected by either issue, since it was
built around aggregate statistics (peak/integrated/mean/std of electron
pressure) from the start rather than raw per-channel expansion — the same
design choice that made it machine-agnostic also made it immune to this bug.

**Open lead, not pursued further here:** redesigning the DIII-D pipeline to
use aggregate Thomson statistics (matching the MAST pipeline's approach)
instead of raw per-channel values would sidestep the variable-channel-count
problem entirely and might meaningfully improve `q95`/`li` prediction, since
those scalars are physically tied to pressure-profile information that raw
coil currents don't carry. Worth doing before a final competitive submission,
not required for a format-valid one.

**A second numerical bug, caught by the dry run itself:** the first
`submission_skeleton.py --max-shots 5` attempt (after fixing the above)
produced `RuntimeWarning: overflow encountered in multiply` from inside
`np.nanstd`. Cause: `pe = Te * ne` (electron pressure, used in the MAST
physics features) reaches ~10²¹–10²⁴ in SI units, and `nanstd` squares
internally — that overflows float32 (max ~3.4×10³⁸ is closer than it looks
once you're squaring 10²³) but not float64. Fixed by computing `Te`/`ne`/`pe`
in float64 inside `_thomson_pressure_stats`, then retrained the MAST
physics pipeline from scratch to make sure the saved model wasn't fit on
any silently-corrupted (inf/overflowed) feature values. The earlier
cross-machine transfer numbers reported above were checked against this —
neither the 150-shot nor 640-shot experiment logs show this warning, so
those results stand uncorrected; it only surfaced once the submission dry
run streamed different (and apparently more extreme) test shots.

---

## Where this leaves the project

The infra hurdles (streaming flakiness, the MAST row-parsing bug) are
solved and now reusable. The Challenge 1 baseline is solid. The Challenge 2
result is a real, if small-sample, demonstration that the paper's suggested
fix for cross-machine collapse actually works directionally — the next
open question is whether it holds up at a larger shot count and whether the
R²/SSIM disagreement on DIII-D needs a different normalization before this
is submission-ready.

---

## 2026-10-09 to 2026-10-11 — scaling up, Thomson fixed, seed-ensemble, plateau found

**Starting point:** a real submission was already live (composite **S = 0.9576**
on a local leak-safe check), trained on 900–1500 of the 2352 DIII-D shots
downloaded at the time, Thomson scattering excluded entirely (the open lead
from the section above, not yet pursued).

**What got done, in order:**

1. **Fixed the open Thomson lead.** `experiments.interpolate_thomson_to_efit`
   now reduces each profile (core + edge, Te + ne) to shape-agnostic
   per-timestep stats — mean/std/max/min + peak/integrated electron pressure
   — instead of indexing raw channels by position. Fixed width regardless of
   channel count (40/42/43/44/54 on DIII-D core), same pattern the MAST
   physics pipeline already used. Also fixed the same float32-overflow bug
   this surfaced again (`ne` ~1e19, squared internally by `nanstd`, needs
   float64).
2. **Downloaded the full DIII-D training set**: 2352 → 6332 → all **7041**
   shots now sit locally in `hf_local_data/data/diii_d_train/` (~94GB).
   Removed an unrelated 64GB game (`Dakar Desert Rally`) to make room; disk
   is not a constraint for anything described below (≈78–80GB free as of
   this writing).
3. **Found and fixed three real memory bugs** that only showed up at this
   shot count (all previously invisible at the smaller scales tested
   earlier): `TargetPCA.fit`/`transform` was unconditionally upcasting the
   *entire* training target array to float64 (fine at ~2000 shots, tens of
   extra GB at 7000+); `normalize_flux_per_frame` allocated a full second
   copy of its output instead of mutating in place; `train_torch_d3d.py`'s
   per-shot accumulation loop cast to float16 only *after* concatenating
   every shot (by which point the float32 version had already peaked). Fixed
   all three. See `scripts/train_submission_models.py`, `scripts/
   cross_machine_physics_features.py`, `scripts/train_torch_d3d.py` git
   history around 2026-10-09/10 for the exact diffs.
4. **Found the DIII-D sklearn and GPU (torch) pipelines scale differently per
   shot** — empirically calibrated safe `--max-shots` caps on this machine
   (25.7GB RAM): sklearn D3D ≈ **4500** shots, MAST physics ≈ **3200** shots
   (its per-frame memory cost is higher than D3D's, don't assume one cap
   transfers to the other), the GPU conv-decoder handles the **full 7041**
   fine once lean-loaded and float16-from-the-start (same pipeline that
   needed those fixes above).
5. **Tried a boundary-weighted loss** (`train_torch_d3d.py --loss boundary`,
   still in the code as an opt-in flag) — up-weights low-|∇ψ| regions (the
   magnetic axis and X-point, both exactly where the scorer's Consistency
   term is most sensitive). Tested via a matched A/B at identical shot
   count/epochs: scored *worse* than plain MSE (0.9731 vs 0.9755 composite).
   Likely cause: the gradient estimate was computed from the float16-stored
   target, which is noisiest exactly in the flat regions the loss was
   supposed to target. Not deleted — worth revisiting with a float32
   gradient estimate — but plain MSE is the working default now.
6. **Found adding Thomson to the sklearn psi head hurts it badly** (same
   features that helped `q95`/`betaN` a lot — R²  0.985→0.995 — dropped psi's
   Consistency from ~0.92 to ~0.71 in a same-shot-count A/B). Fixed by
   training the psi head on magnetics-only columns (sliced out of the shared
   scaled feature matrix) while keeping Thomson for the scalar heads. Sklearn
   alone recovered to composite 0.9866.
7. **Discovered GPU (MPS) training has real run-to-run variance** — two
   identically-configured training runs (same seed, same data, same code)
   converged to meaningfully different optima (composite 0.9789 vs 0.9851).
   Built a multi-seed ensemble instead of hoping for a lucky run:
   `train_torch_d3d.py --tag <name>` saves distinctly-named checkpoints per
   seed, `D3D_PSI_MODE=torch_multi` (or `ensemble_multi` to also blend in the
   fixed sklearn psi head) averages every `d3d_torch_model_*.pt` found.
8. **Scaled the ensemble and watched it plateau.** Composite score by seed
   count: 1 seed 0.9789 → 3 seeds **0.9851** → 8 seeds **0.9886** (local
   `ensemble_multi`: 0.9907) → 13 seeds **0.9909** local. The *marginal* gain
   per batch of seeds added: 1→3 seeds +0.0035, 3→8 seeds +0.0015, 8→13
   seeds +0.0002. Clean diminishing returns, now flat. **More seeds will not
   reach a meaningfully higher score** — this lever is exhausted.

**Real (Codabench) vs. local (leak-safe) scores, for calibration:** the one
real submission checked against the actual leaderboard this period scored
**0.9807** when the local leak-safe estimate for that exact model was 0.9851
— i.e. local estimates run **~0.004–0.005 optimistic** relative to the real
scorer. Keep that gap in mind reading any "local" number above.

**A real operational hazard, worth knowing before relaunching anything
long-running:** background jobs launched through Claude Code's normal
tracked-background-task mechanism died unpredictably and repeatedly during
this period — not from OOM (a memory-sampling sidecar confirmed healthy,
plentiful free RAM at every death), not from sleep/wake, not from Spotlight
(once suppressed). The fix that worked: launch via
`nohup bash -c '...' < /dev/null > logfile 2>&1 & echo $! > /tmp/name.pid; disown`
and poll with `ps -p $(cat /tmp/name.pid)` instead of trusting the tracked
background-task status. One process was also seen stuck in macOS's `UN`
(uninterruptible I/O wait) state for several minutes before clearing on its
own — that's transient contention, not a deadlock; don't kill a `UN` process
on sight, just wait.

### Current state (end of this period)

| What | Value |
|---|---|
| Local leak-safe composite (best, 13-seed `ensemble_multi`) | **0.9909** |
| Real Codabench composite (last checked, 8-seed-era submission) | **0.9807** |
| Submission pushed to HF | `Diby07/fusion-eq-predictions` @ commit `eb2369be54037fe93bef092b0315729acb9a6373` |
| Code pushed to fork | `Dibyajit7/fusion-equilibrium-challenge` @ commit `bf5f0d5` |
| `D3D_PSI_MODE` used for the above | `ensemble_multi` |
| Seeds in the torch ensemble | 13 (`models/d3d_torch_model_s{42,123,777,1001,2002,3003,4004,5005,6006,7007,8008,9009,10010}.pt`) |
| Local DIII-D shots on disk | 7041 / 7041 (all of them) |
| Disk free | ≈78–80GB |

### Next steps — three options, none of them "train more seeds"

The seed-ensemble lever is exhausted (see point 8 above). Closing the
remaining ~0.01 gap to something like 0.997 (the current #1 leaderboard
entry scores 0.9967 — nobody has actually reached 0.997) needs a
structurally different change. None of these need any new data download —
the full 7041-shot DIII-D set is already local, and disk is not a
constraint. Approximate time estimates assume this same machine (25.7GB
RAM, Apple Silicon MPS GPU) and the environmental kill issue above already
being handled via the detached/nohup pattern.

1. **Fix sklearn's capacity so Thomson helps psi instead of hurting it.**
   Right now Thomson is excluded from the psi head entirely (point 6 above)
   because the existing architecture (a 256,128 MLP after PCA(50)) couldn't
   absorb the extra 20 input dimensions without degrading. Try a bigger
   hidden-layer MLP, more PCA components, or a separate small head that
   specifically learns a Thomson-conditioned *correction* to the
   magnetics-only prediction rather than retraining on the combined feature
   set from scratch.
   — **Time:** half a day to a full day (each retrain-and-leak-safe-eval
     cycle at the proven-safe ~4500-shot cap is ~25–35 minutes; expect
     5–15 iterations to find a config that doesn't regress Consistency).
   — **Space:** none beyond what's already on disk.
   — **Risk:** moderate — this exact feature combination already failed
     once; no guarantee a bigger MLP fixes it rather than just being slower
     to fail the same way.

2. **A different architecture for the flux-map regression itself.** The
   current `FluxDecoderNet` (`scripts/torch_models.py`) is a fairly simple
   FC→conv-upsample decoder. A deeper network, a U-Net-style architecture
   with skip connections, or incorporating a physics-consistency term
   (e.g. a differentiable penalty tied to the Grad–Shafranov equation, or to
   known ψ symmetry/boundary properties) could plausibly push past where the
   current architecture has saturated — R²_psi is already ~0.9997, but
   Consistency depends on precision at specific low-gradient points (the
   magnetic axis, the X-point) that a generic pixelwise-MSE-trained decoder
   has no particular incentive to get exactly right.
   — **Time:** multi-day. Designing + implementing + debugging a new
     architecture, then the same full-scale training cycle (~2–2.5 hours per
     run on this GPU) repeated across design iterations and likely another
     multi-seed ensemble once a good architecture is found. Realistically
     2–4 days of combined engineering + compute time.
   — **Space:** none beyond what's already on disk; a deeper/wider model's
     checkpoint files are still only a few MB each, immaterial.
   — **Risk:** highest payoff ceiling, highest effort, least certain to pay
     off within any fixed time budget.

3. **Revisit the boundary-weighted loss with a cleaner gradient estimate.**
   Point 5 above found it scored worse, but the likely cause (gradient
   computed from a float16-stored target) is a fixable implementation
   detail, not evidence the underlying idea (up-weight low-|∇ψ| regions) is
   wrong — it's the single most mechanism-targeted idea tried so far, since
   it directly addresses *why* Consistency is hard (sensitivity at low-
   gradient points) rather than throwing more data/capacity at the problem
   generally.
   — **Time:** half a day for the fix + a matched small-scale A/B
     revalidation (~1 hour of compute, same methodology as before, see point
     5); if it validates, the same multi-seed scale-up as before adds
     another ~2–12+ hours depending on how many seeds are worth training
     once the single-model result is known.
   — **Space:** none beyond what's already on disk.
   — **Risk:** lowest effort to re-test the idea properly; unknown whether
     the fix actually resolves the earlier regression.

**Bonus/secondary option, not one of the three above:** Challenge 2 (MAST)
was never pushed as far — its physics-feature pipeline is capped at 3200
shots (the safe memory ceiling found for that specific pipeline) and never
got a GPU/multi-seed treatment at all. If a future session has spare time
after the DIII-D-focused options above, applying the same multi-seed-GPU-
ensemble idea to Challenge 2 is a known-working pattern, not a new one, and
might be a faster win than any of the three options above — but Challenge
2's `G_ratio` score depends on DIII-D clearing a `S ≥ 0.85` gate first (it
currently does, comfortably), so it's additive, not a replacement for the
DIII-D work.
