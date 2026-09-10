# ADR 0020 — Supersample factor, box width and the optics stage order

**Status:** Accepted
**Date:** 2026-09-11

## Context

§8.3 says: render at 3–4× and downsample with a box matching the pixel footprint, which yields the
detector MTF, aliasing and sub-pixel target behaviour with no extra model. Two details are open in the
spec: whether the box is the pixel *pitch* or the *active width* √fill · pitch (spec issue S21 —
§8.3's worked MTF_det(ξ_N) ≈ 0.64 assumes w = pitch; the 0.90 fill factor would give 0.669), and in
which order the optics terms apply.

## Options considered

- **Box = pitch, fill factor only in A_d** — the detector integrates over its full footprint in the
  image plane for MTF purposes (the inactive border is small and the sample spacing is the pitch);
  reproduces §8.3's 0.64; one integer factor, exact block mean.
- **Box = active width** — a fractional-width sub-box needs edge weights; 4 % more MTF at Nyquist;
  no measurement to prefer it.
- **Blur-then-decimate** (Gaussian at native, then pick samples) — loses the aliasing that corrupts
  small targets; explicitly what §8.3 warns against.

## Decision

- **Box width = pitch; fill factor scales A_d** (ADR 0017's `active_width_um` remains the number for
  any future analytic MTF_det). `box_downsample` is an exact k×k block *mean* (units unchanged).
- **Supersample factor k = 4 by default** (`optics.supersample_factor`, 1–8). On a discrete k× grid
  the box transfer is the Dirichlet kernel sin(πξ)/(k sin(πξ/k)), 2.6 % above |sinc| at Nyquist for
  k = 4; tests check the exact form and the sinc limit at k = 32.
- **Order** in `apply_optics`: (1) optical PSF at the supersampled pitch [identity until MS], (2) box
  downsample, (3) × π τ_opt/(4F² + 1) · cos⁴ · A_d, (4) + Φ_self. Vignetting after the downsample
  (cos⁴ varies by < 1e-4 across a pixel, tested in M3.2). `invert_optics` undoes (3)–(4) including
  cos⁴, so the radiometric branch recovers scene radiance at every pixel.
- The G-buffer arrives already supersampled from the engine (ADR 0014: ids never blend; radiance is
  what gets box-filtered), so stage 1 runs at k× and stage 3 returns to native.

## Consequences

A k× render costs k² the pixels; the NumPy oracle at 4× on 640×512 is 5.2 Mpx per stage, seconds
per frame. Sub-pixel targets (MS.6) see exactly the box/aliasing physics of §8.3. The 2.6 % Dirichlet
vs sinc difference at k = 4 is the accepted discretisation error of the detector MTF.

## Revisit when

A measured slant-edge MTF for the Boson disagrees with the sinc(pitch) model by more than the
aberration Gaussian can absorb, or sub-pixel target work needs k > 4.
