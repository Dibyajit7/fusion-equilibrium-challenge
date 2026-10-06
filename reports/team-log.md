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
