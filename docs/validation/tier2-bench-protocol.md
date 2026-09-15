# Tier 2 bench protocol — for when a camera arrives

Roadmap M12.4 · docs/physics-model.md §15 Tier 2 · ADR 0003

## Why this document exists before the camera does

No infrared camera is available to this project. Every Tier 2 number the repository reports today
is a **self-consistency check**: the simulated SITF is linear in `L_B` because the code makes it so,
and the simulated NETD matches its own anchor because that is what an anchor is. None of it is
evidence about a real camera, and the README's limitations say so.

Writing the protocol now, rather than when hardware appears, is deliberate. Deciding a tolerance
while looking at the first measurement is how a bench becomes a rubber stamp. Everything below —
the file layout, the fitting rule, the pass criteria — is fixed in advance, and
`irsim.validation.measured` implements it, so the day a camera arrives the benches already know how
to read it.

## The four measurements

### 1. SITF — signal transfer function

**What.** Mean DN against blackbody temperature, over the range the camera will be used in.

**How.** An extended blackbody filling the field, at a stand-off short enough that the atmospheric
path is negligible (< 0.5 m) or with the path measured and divided out. Ten setpoints from 253 K to
453 K is the shape the simulated bench uses (M3.12). At each setpoint: let the source settle to
within 0.05 K, then average **100 frames** and read the mean over a centre ROI covering 25 % of the
array — the ROI keeps `cos⁴` vignetting out of the number, which is a lens property and not a
transfer property.

**Record.** `data/validation/sitf/<camera>.csv`, columns `T_blackbody_K, DN`.

**Pass.** Shape only, after fitting a gain and offset: residual rms < **1 % of the measured DN
span**. Absolute DN is a range choice (ADR 0019), so requiring it to match would be requiring the
simulator to guess the bench's gain. What is being tested is that the *curvature* — which comes
from Planck through the band, and from nothing else — is right.

### 2. NETD

**What.** Noise-equivalent temperature difference in millikelvin, at 300 K and at a second, higher
setpoint.

**How.** Two blackbody setpoints 5–10 K apart. Take 100 frames at each; NETD = σ_temporal ·
ΔT / ΔDN, with σ_temporal the per-pixel temporal standard deviation averaged over the ROI. Take the
second setpoint at **373 K** as well: a bolometer's NETD falls as the scene warms because ∂L/∂T
rises, and a bench that measures one temperature cannot see that.

**Record.** `data/validation/netd/<camera>.csv`, columns `T_blackbody_K, NETD_mK`.

**Pass.** Absolute, **no fit**: within **15 %**, *and* the 373 K value below the 300 K value. A
gain fit here would let the model be wrong by any factor and still pass, which is the one thing
this measurement exists to rule out.

### 3. Three-dimensional noise

**What.** The NVESD seven — σ_T, σ_V, σ_H, σ_TV, σ_TH, σ_VH, σ_TVH — in DN.

**How.** A uniform blackbody at 300 K, **200 frames**, camera and source both thermally settled,
no FFC during the capture (or the FFC frames removed — a shutter event inside the cube appears as
an enormous σ_T). Decompose with `irsim.validation.noise.decompose_3d`, which is the same estimator
the simulated bench uses, so the comparison is of cameras and not of estimators.

**Record.** `data/validation/noise3d/<camera>.csv`, columns `component, sigma_DN`, one row per
component, component names lowercase (`tvh`, `vh`, `h`, …).

**Pass.** Shape only, after a gain fit — the DN scale is a range choice as above — with residual
rms < **10 % of the σ_TVH value**, *and* every component above its own sampling floor
(`estimate_floors`). A component below its floor is reported as unmeasured, not as zero: ADR 0023's
rule, and the same one ME.5 applied to the public data.

### 4. MTF — slant edge

**What.** Modulation transfer function against spatial frequency, to the detector Nyquist.

**How.** A knife edge against a warm background, tilted **4°–6°** from a column (not 45°: the
projection is a sinc that is hardest to invert exactly there, and not 0°, which gives no sub-pixel
sampling at all). Average 32 frames to suppress temporal noise, then `irsim.validation.mtf.
slant_edge_mtf`.

**Record.** `data/validation/mtf/<camera>.csv`, columns `frequency_cyc_per_mm, MTF`.

**Pass.** Absolute, **no fit**: within **0.05 in MTF** at every frequency up to Nyquist. MTF is
already a ratio normalised to 1 at DC, so there is no scale left to fit and a discrepancy is a
discrepancy.

## What a passing bench would and would not establish

It would establish that the **detector and optics chain** is right: the transfer curvature, the
noise magnitude and its spatial structure, the resolution. It would say nothing about the
radiometry of a *scene* — emissivity, the atmosphere, the thermal solver — which is Tier 3 and
Tier 4, and which needs a scene, not a blackbody.

It would also not close ADR 0003. The sim-to-real question this project exists to answer is about
imagery of targets against sky, and a laboratory blackbody has no sky in it.

## Running it

The comparisons are **skip-if-absent**: with no file under `data/validation/<quantity>/`, the Tier 2
tests run their self-consistency half and skip the comparison with a reason naming the path they
looked for. Drop the CSV in and the comparison runs with no other change. `measured_path` is the one
place the layout is written down, so a future bench cannot invent a second one.
