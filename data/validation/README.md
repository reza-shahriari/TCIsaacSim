<!-- GENERATED FILE - do not edit.
     Source: data/validation/datasets.yaml
     Regenerate: python scripts/fetch_validation_data.py --render-readme
     tests/unit/test_validation_manifest.py fails if this file is stale. -->

# Validation data

No IR camera is available to this project (ADR 0003), so every comparison against reality
is made against imagery somebody else published. What each set *is* therefore decides
what may be measured on it:
a three-dimensional noise decomposition means something on a Y16
stream and nothing on an unknown ISP's output through a lossy codec, and a box size
converts to a range only when the sensor's pitch and focal length are known.

The datasets themselves are **not** in git. `scripts/fetch_validation_data.py` obtains or
verifies them into `data/validation/<set>/`.

**Primary set: `halmstad_drone_detection`** -- A dataset for multi-sensor drone detection (Svanström et al. 2021).

The only set here with an unambiguous licence, a citable DOI, and a fully documented signal path from a camera core this project already models. That combination makes it the reference set: every other entry is a supplement to it, not a substitute.

## Licences

`unstated` means the publisher did not state terms. That is not a guess that they are
permissive: it is a record that they are unknown, and the fetch script refuses such a set
unless asked explicitly, so using data on unknown terms is always a deliberate act.

| set | licence | access | role |
|---|---|---|---|
| `halmstad_drone_detection` | CC0-1.0 | manual | primary |
| `anti_uav_410` | unstated | manual | supplement |
| `cst_anti_uav` | unstated | unreleased | supplement |
| `lrddv3` | unstated | manual | supplement |
| `irstd_1k` | unstated | manual | prior |
| `nuaa_sirst` | unstated | manual | prior |

Fields last checked against the publishers' own pages: **2026-09-13**.

## `halmstad_drone_detection`

**A dataset for multi-sensor drone detection (Svanström et al. 2021)**

The only set here with an unambiguous licence, a citable DOI, and a fully documented signal path from a camera core this project already models. That combination makes it the reference set: every other entry is a supplement to it, not a substitute.

| field | value |
|---|---|
| licence | CC0-1.0 |
| licence note | CC0-1.0 is the licence on the code repository, which also states the data is "free to download, use and edit". UNVERIFIED whether the Zenodo record carries the same terms; confirm at the DOI before redistributing any frame. |
| access | manual |
| doi | 10.5281/zenodo.5500576 |
| paper | 10.1016/j.dib.2021.107521 |
| code | https://github.com/DroneDetectionThesis/Drone-detection-dataset |
| sensor | FLIR Breach PTQ-136 (Boson 320x256 core) |
| resolution | [320, 256] |
| field of view (deg) | [24.0, 19.0] |
| frame rate (Hz) | 60.0 |
| bit depth (native / stored) | 16 / 8 |
| codec | mp4 |
| clips | 365 |
| clip length (s) | 10.0 |
| annotated frames | 203328 |
| classes | drone, bird, airplane, helicopter |
| label format | matlab .mat (a python decoder ships in the repo) |
| sha256 | not downloaded |

**Signal path.** Boson Y16 16-bit stream -> recorder converts to 8-bit -> stored as mp4. The frames therefore carry the core's noise, striping and FFC freezes but NOT its AGC, DDE or palette: the recorder's conversion stands in for the ISP and its rule (linear? min-max per frame?) is itself unverified. Treat any histogram-shape statistic as measuring the recorder.

**May be used for:** `noise_3d`, `spatial_psd`, `temporal_psd`, `ffc_freeze`, `fixed_pattern_growth`, `bad_pixels`, `target_size_and_scr`, `edge_spread`, `smear`.

**Must not be used for:** `agc_signature`, `dde_overshoot` -- see the signal path above.

Ten-second clips are too short for an FFC *interval* distribution (the Boson's is 180 s); they can still show freeze length. Maximum sensor-to-target range is 200 m by the flight regulations in force, so this set says nothing about the 0.5-5 km regime phase 1 models.

## `anti_uav_410`

**Anti-UAV410 (Huang et al., TPAMI 2023)**

410 sequences long enough to show an FFC *interval* distribution, which the 10-second Halmstad clips cannot. Its frames are display output, so it is the set that says what a finished thermal picture of a drone looks like.

| field | value |
|---|---|
| licence | unstated |
| licence note | The repository states no licence for the data. Unstated is not permissive. Using the frames for measurement on one machine is a different question from redistributing them or training a model that is shipped; settle that before either. |
| access | manual |
| paper | 10.1109/TPAMI.2023.3335338 |
| code | https://github.com/HwangBo94/Anti-UAV410 |
| sensor | undocumented |
| bit depth (native / stored) | None / 8 |
| codec | lossy |
| clips | 410 |
| annotated boxes | 438000 |
| label format | per-sequence json boxes |
| sha256 | not downloaded |

**Signal path.** The camera's display output through an unknown ISP, stored lossy. AGC, DDE, palette and any temporal filtering are all baked in and none of them are documented, so nothing measured here can be attributed to the sensor rather than to its ISP.

**May be used for:** `ffc_interval`, `agc_signature`, `dde_overshoot`, `target_size_and_scr`, `contrast_polarity`.

**Must not be used for:** `noise_3d`, `temporal_psd` -- see the signal path above.

The sensor being undocumented is the binding constraint: with no pitch, focal length or NETD there is no way to convert a measured box size into a range or a measured contrast into kelvin. Use it for shape-of-the-picture statistics only.

## `cst_anti_uav`

**CST Anti-UAV (Xie et al., ICCV 2025 Workshops)**

220 sequences built specifically around *tiny* targets in complex scenes -- the regime where the MS.6 point-target path replaces the renderer, and the one this simulator is least able to check any other way.

| field | value |
|---|---|
| licence | unstated |
| access | unreleased |
| paper | arXiv:2507.23473 |
| code | https://github.com/PCwenyue/CST-Anti-UAV |
| sensor | undocumented |
| bit depth (native / stored) | None / 8 |
| codec | lossy |
| clips | 220 |
| annotated boxes | 240000 |
| sha256 | not downloaded |

**Signal path.** Display output, as Anti-UAV410. UNVERIFIED in detail; confirm from the paper before use.

**May be used for:** `target_size_and_scr`, `contrast_polarity`, `cloud_clutter_psd`.

**Must not be used for:** `noise_3d`, `agc_signature` -- see the signal path above.

Listed now so the index is complete and so nobody re-derives its status later. Check the repository before planning work against it.

## `lrddv3`

**LRDDv3: long-range drone detection with range information and thermal data**

The only indexed set with per-image **range** labels, which is the one thing needed to test size-versus-range and SCR-versus-range against reality rather than against the model's own geometry.

| field | value |
|---|---|
| licence | unstated |
| licence note | Access requires a request form and is stated to be subject to US export control regulations. That is an access restriction, not a licence; read the form's terms before use and do not assume the data may be redistributed or published from. |
| access | manual |
| paper | arXiv:2605.25942 |
| site | https://research.coe.drexel.edu/ece/imaple/lrddv3/ |
| sensor | undocumented thermal, 640x512 |
| resolution | [640, 512] |
| frame rate (Hz) | 5.0 |
| bit depth (native / stored) | None / 8 |
| clips | 128 |
| thermal images | 29630 |
| sha256 | not downloaded |

**Signal path.** Frames sampled at 5 fps from video. UNVERIFIED whether the thermal frames are Y16-derived or display output. At 5 fps nothing temporal can be measured: no FFC freeze, no temporal PSD.

**May be used for:** `target_size_and_scr`, `size_vs_range`.

**Must not be used for:** `noise_3d`, `temporal_psd`, `ffc_freeze`, `smear` -- see the signal path above.

Ranges are mostly 0-50 m and reach only ~175 m, and the geometry is air-to-air with ground clutter behind the target. It therefore constrains the near end of the range law and says nothing about a 0.5-5 km sky-background target, which is the case phase 1 actually models.

## `irstd_1k`

**IRSTD-1k infrared small-target detection set**

A thousand single frames of small targets against sky, sea and land. No sequence, no sensor, no radiometry -- but a large sample of what target *size* and local contrast look like.

| field | value |
|---|---|
| licence | unstated |
| access | manual |
| sensor | undocumented |
| resolution | [512, 512] |
| bit depth (native / stored) | None / 8 |
| frames | 1000 |
| sha256 | not downloaded |

**Signal path.** unknown; single frames, provenance not documented per image

**May be used for:** `target_size_prior`, `scr_prior`.

**Must not be used for:** `noise_3d`, `temporal_psd`, `ffc_freeze`, `agc_signature`, `size_vs_range` -- see the signal path above.

Priors only. With no sensor and no range there is nothing to convert a pixel count into, so a statistic from this set can bound a distribution but can never be a target the simulator is tuned to hit.

## `nuaa_sirst`

**NUAA-SIRST (SIRST v1) single-frame infrared small-target set**

As IRSTD-1k, and commonly reported alongside it, so the two are indexed together.

| field | value |
|---|---|
| licence | unstated |
| access | manual |
| sensor | undocumented |
| bit depth (native / stored) | None / 8 |
| frames | 427 |
| sha256 | not downloaded |

**Signal path.** unknown; single frames, provenance not documented per image

**May be used for:** `target_size_prior`, `scr_prior`.

**Must not be used for:** `noise_3d`, `temporal_psd`, `ffc_freeze`, `agc_signature`, `size_vs_range` -- see the signal path above.

Priors only, for the same reasons as IRSTD-1k.
