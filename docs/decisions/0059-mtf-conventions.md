# ADR 0059 — MTF conventions: supersample box as MTF_det, band-representative λ, one Gaussian, motion MTF photon-only

**Status:** Accepted
**Date:** 2026-09-11

## Context

§8.3 lists six MTF factors and a worked case (12 µm, F/1, 10 µm: MTF_diff(ξ_N) ≈ 0.49, MTF_det(ξ_N)
≈ 0.64) but leaves open: which λ to use for diffraction in a 6 µm-wide band, whether MTF_aberr,
MTF_defocus and MTF_elec are separate models (spec issue S22), how MTF_det relates to the supersample
box filter of ADR 0020, and whether motion MTF applies to a bolometer whose smear is already the τ_th
IIR (spec issue S20).

## Decision

`irsim.optics.mtf`, `irsim.optics.psf`:

- **MTF_det is the supersample box filter.** The pipeline never applies a detector PSF; the k×
  render followed by the k×k box mean *is* MTF_det with the aliasing (ADR 0020). `mtf_detector` exists
  for the analytic cascade and the bench only, and `mtf_system` multiplies it in only when asked.
- **Optical PSF = MTF_diff · MTF_gauss**, synthesised at the supersampled pitch as the inverse FFT
  of the radially symmetric transfer, normalised to Σ = 1 (tiny FFT ringing clipped), applied by FFT
  convolution with edge replication before the box downsample.
- **Band-representative λ for diffraction** = the R-weighted mean wavelength of the response
  (`SpectralResponse.mean_wavelength_um`, 10.44 µm for the Boson estimate), or
  `mtf.reference_wavelength_um` when set, else the band centre. The λ-dependence of the diffraction MTF
  across a 6 µm band is a second-order effect at ξ_N (≈ ±10 % of MTF_diff) and is accepted.
- **One Gaussian** (`mtf.aberration_sigma_um`, fitted from a slant edge) stands for aberration and
  defocus; **MTF_elec = 1**.
- **Motion MTF is photon-detector only** (sinc of the smear during t_int); bolometer smear is the
  inter-frame IIR of §9.2 (M9.1), never a second blur.
- `np.sinc` is normalised: the argument is w·ξ. The double-π bug (0.198 instead of 0.637 at Nyquist)
  is pinned by a test.

## Consequences

For the Boson at F/1, 10.5 µm, 12 µm pitch: MTF_diff(ξ_N) = 0.485, MTF_det(ξ_N) = 0.637, system 0.31
at Nyquist, measured by the slant-edge bench within 0.05. The PSF kernel at 4× is 33×33 samples and
the FFT convolution of a 4× frame costs a few hundred ms in NumPy. The pipeline's exact-box tests run
with `psf_enabled=False`.

## Revisit when

ME.3's edge-spread measurements from public video show a wider or asymmetric halo than diffraction +
one Gaussian, or a photon-band camera with a documented electronic MTF is modelled.
