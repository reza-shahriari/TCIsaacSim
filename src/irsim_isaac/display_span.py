"""The manual display span for a filmed sequence, in whichever quantity the band has (M10.23).

**Why a manual span at all.** Both of §11.3's AGC modes rescale themselves from the current
frame's own histogram, so a target whose temperature is the subject comes out looking the same in
every frame: the gain follows the target and cancels exactly the change being filmed. A real
operator switches to a manual span for that, and so do these renders. The camera's own AGC output
is filmed alongside, so a viewer can see both.

**Why this module exists.** The span used to be written in kelvin, taken from the target's own
thermal nodes and applied to the ``apparent_t`` output. That is right for an emissive band and
**impossible** in a reflective one: §12.1 turns ``apparent_temperature`` off for SWIR and NIR
because inverting L_B does not give a scene temperature when the signal is reflected sunlight, so
there is no plane to span and no kelvin to span it in. A NIR render crashed on exactly that.

The span is therefore chosen per band:

* an **emissive** band spans ``apparent_t`` between two temperatures, as before, and the numbers
  in the overlay are the temperatures a viewer should read off the picture;
* a **reflective or mixed** band spans the raw ``dn16`` between two percentiles of the *first*
  captured frame, held fixed for the sequence. Percentiles of one frame rather than of each frame
  is the whole point -- it is a manual span, not a slow AGC -- and the first frame is a fair
  choice in these bands because the picture is reflected sunlight, which does not change while a
  motor warms up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SpanKind",
    "DisplaySpan",
    "span_from_nodes",
    "span_from_dn16",
    "span_from_apparent_t",
]

SpanKind = Literal["apparent_t", "dn16"]


@dataclass(frozen=True)
class DisplaySpan:
    """Two levels and the plane they apply to. ``scale`` returns 0-1 for the palette."""

    kind: SpanKind
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError(f"a display span must increase: got {self.low} to {self.high}")

    @property
    def is_temperature(self) -> bool:
        return self.kind == "apparent_t"

    @property
    def caption(self) -> str:
        if self.is_temperature:
            return f"span {self.low - 273.15:.0f}-{self.high - 273.15:.0f}C"
        return f"span DN {self.low:.0f}-{self.high:.0f}"

    def plane(self, outputs: Any) -> NDArray[np.float64]:
        """The output plane this span applies to, as float64.

        Raises rather than falling back: a band whose config switched `apparent_temperature` off
        and a span that still asks for kelvin is a configuration error, and rendering the other
        plane instead would produce a plausible picture of the wrong quantity.
        """
        value = outputs.apparent_t if self.is_temperature else outputs.dn16
        if value is None:
            missing = "apparent_t" if self.is_temperature else "dn16"
            raise ValueError(
                f"the span is in {self.kind} but the camera produced no {missing} plane; "
                f"enable outputs.{'apparent_temperature' if self.is_temperature else 'dn_16'} "
                "or choose the other span"
            )
        return np.asarray(value, dtype=np.float64)

    def scale(self, outputs: Any) -> NDArray[np.float64]:
        return (self.plane(outputs) - self.low) / (self.high - self.low)

    def code_for(self, value: float) -> int:
        """Where one level lands on the 0-255 display axis, for the printed key."""
        return max(0, min(255, round(255.0 * (value - self.low) / (self.high - self.low))))


def span_from_nodes(node_samples: Any, *, below: float = 0.25, above: float = 0.25) -> DisplaySpan:
    """An emissive band's span, from the target's own thermal nodes over the sequence.

    Delegates to :func:`irsim_eval.video.target_span_k`, which is where the padding rule lives, so
    there is one definition of "the span the target needs" rather than two.
    """
    from irsim_eval.video import target_span_k

    low, high = target_span_k(node_samples, below=below, above=above)
    return DisplaySpan(kind="apparent_t", low=float(low), high=float(high))


def span_from_dn16(
    frame: Any, *, low_pct: float = 1.0, high_pct: float = 99.5, pad: float = 0.05
) -> DisplaySpan:
    """A reflective band's span, from percentiles of one frame's raw ADC plane.

    The percentiles clip the tails a single hot specular pixel would otherwise own; ``pad`` widens
    the result slightly so the brightest real structure is inside the span rather than exactly on
    its edge. If the frame is flat the span falls back to the full ADC range, which renders a flat
    picture as flat instead of raising on a scene that is simply uniform.
    """
    values = np.asarray(frame, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no pixels to take a span from")
    lo, hi = (float(x) for x in np.percentile(values, [low_pct, high_pct]))
    if hi <= lo:
        # A frame flat enough that both percentiles land on the same value. Fall back around the
        # **median**, not the minimum: a single dead pixel or a glint is exactly what the
        # percentiles were there to exclude, and anchoring the fallback on an extremum would hand
        # the whole span to the outlier the rest of this function just removed.
        middle = float(np.median(values))
        return DisplaySpan(kind="dn16", low=middle - 0.5, high=middle + 0.5)
    width = hi - lo
    return DisplaySpan(kind="dn16", low=lo - pad * width, high=hi + pad * width)


def span_from_apparent_t(
    frame: Any, *, low_pct: float = 1.0, high_pct: float = 99.5, pad: float = 0.05
) -> DisplaySpan:
    """An emissive band's span from percentiles of one frame's apparent temperature.

    The counterpart of :func:`span_from_nodes` for a scene with **no single target node** whose
    temperature is the subject. A maritime frame is the sea and the sky: there is nothing to centre
    a span on, and spanning the thermal nodes of whatever vessels happen to be in shot would throw
    away the sea, which is most of the picture and the thing the angular-emissivity model exists to
    render. Percentiles of the frame itself are the honest choice there.

    Still a *manual* span: it is taken once, from one frame, and held for the sequence, so the
    picture does not breathe the way an AGC's would.
    """
    values = np.asarray(frame, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no pixels to take a span from")
    lo, hi = (float(x) for x in np.percentile(values, [low_pct, high_pct]))
    if hi <= lo:
        # See `span_from_dn16`: a flat frame falls back around the median, which the percentiles
        # already agree on, rather than around an extremum they were there to exclude.
        middle = float(np.median(values))
        return DisplaySpan(kind="apparent_t", low=middle - 0.5, high=middle + 0.5)
    width = hi - lo
    return DisplaySpan(kind="apparent_t", low=lo - pad * width, high=hi + pad * width)
