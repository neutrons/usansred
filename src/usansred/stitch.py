"""Stitch the I(Q) profiles of different analyzer orders into a single profile.

The analyzer reflects several orders (harmonics) of the primary wavelength, and each order is
reduced into its own I(Q) profile. Order ``n`` covers a Q range roughly ``n`` times wider than
order 1, so adjacent orders overlap in Q. This module joins the profiles one after another:
order 1 is the fixed reference, and each higher order is scaled onto the profile stitched so
far, using the points of the two profiles that lie in their Q-range intersection.

The peak of the unscattered beam at low Q is dominated by the instrument resolution, which
differs from order to order, so it is excluded: the scale factor is fitted only over the part
of the overlap above a lower bound ``q_fit_min``, and only the points of the joined order above
that bound are added to the stitched profile. The scale factor is an error-weighted
least-squares fit (see ``scale_factor``).

Momentum transfer Q is in ``1/angstrom`` throughout. The user documentation describes the procedure
in :ref:`stitching the harmonics <stitching-harmonics>`.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from usansred.models import IQData
from usansred.utils.logging import get_logger

__all__ = [
    "MissingOrderError",
    "QRangeNotExtendedError",
    "StitchError",
    "StitchResult",
    "apply_stitch",
    "interpolate_error",
    "interpolate_profile",
    "merge_profiles",
    "overlap_range",
    "scale_factor",
    "stitch_profiles",
    "valid_point_mask",
]

logger = get_logger(__name__)


class StitchError(ValueError):
    """Raised when a profile cannot be joined onto the stitched profile.

    Attributes
    ----------
    log_level : int
        Level at which ``stitch_profiles`` logs the failure, ``logging.ERROR`` by default.
    """

    log_level: int = logging.ERROR


class QRangeNotExtendedError(StitchError):
    """Raised when an order does not extend the stitched profile to higher Q.

    Higher orders are often too weak to carry counts at high Q, so this is an expected outcome
    rather than a fault in the data, and it is logged as a warning (except for order 2, see
    ``stitch_profiles``).
    """

    log_level: int = logging.WARNING


class MissingOrderError(StitchError):
    """Raised by ``apply_stitch`` when an order to be stitched has no data.

    This is the expected outcome when the background has no data for an order that the sample
    has, so it is logged as a warning.
    """

    log_level: int = logging.WARNING


@dataclass
class StitchResult:
    """Outcome of stitching the profiles of several orders.

    Attributes
    ----------
    profile : IQData
        The stitched profile, sorted by Q. Empty when not even order 1 could be used.
    joined_orders : list[int]
        Orders (1-based) included in ``profile``, in the order they were joined.
    scale_factors : list[float]
        Scale factor applied to each order in ``joined_orders``; the first one (order 1) is 1.0.
    overlaps : list[tuple[float, float]]
        Q-range intersection ``(q_min, q_max)``, in ``1/angstrom``, of each join. Entry ``k``
        belongs to the join of ``joined_orders[k + 1]``, so there is one fewer entry than orders.
    fit_ranges : list[tuple[float, float]]
        Q range ``(max(q_min, q_fit_min), q_max)``, in ``1/angstrom``, over which the scale factor
        of each join was fitted. Its lower end is also the lowest Q at which points of the joined
        order are added to the stitched profile. Indexed like ``overlaps``.
    masks : list[np.ndarray]
        Boolean mask of the valid (non-zero-count) points of each order in ``joined_orders``,
        over that order's input profile.
    """

    profile: IQData = field(default_factory=IQData)
    joined_orders: list[int] = field(default_factory=list)
    scale_factors: list[float] = field(default_factory=list)
    overlaps: list[tuple[float, float]] = field(default_factory=list)
    fit_ranges: list[tuple[float, float]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)


def interpolate_error(q: float, q_ref: np.ndarray, e_ref: np.ndarray) -> float:
    r"""Interpolate an uncertainty at ``q`` by propagating variances in quadrature.

    Linear interpolation of the intensity between the two bracketing points is
    ``I = (1 - w) * I0 + w * I1``, with ``w`` the fractional position of ``q`` in the interval.
    Treating the two measurements as statistically independent, the uncertainty of that
    combination is

    .. math::

        \sigma = \sqrt{\left[(1 - w)\,\sigma_0\right]^2 + \left[w\,\sigma_1\right]^2}

    The interpolation weights must be squared. Interpolating the variance with the plain
    weights, ``sqrt((1 - w) * e0**2 + w * e1**2)``, is a different and also incorrect result.
    Note that the correct value can fall below both endpoint uncertainties; combining two
    independent measurements does reduce the variance.

    Outside the range of ``q_ref`` the nearest endpoint uncertainty is returned, which matches
    how ``numpy.interp`` clamps the intensity and is the ``w = 0`` / ``w = 1`` limit of the
    formula above.

    Parameters
    ----------
    q : float
        Momentum transfer at which the uncertainty is needed, in ``1/angstrom``.
    q_ref : np.ndarray
        Momentum-transfer values of the profile being interpolated, in ``1/angstrom``,
        sorted in ascending order.
    e_ref : np.ndarray
        Uncertainties, one per entry of ``q_ref``.

    Returns
    -------
    float
        The interpolated uncertainty at ``q``.
    """
    hi = int(np.searchsorted(q_ref, q))
    if hi == 0:  # q below the range
        return float(e_ref[0])
    if hi == len(q_ref):  # q above the range
        return float(e_ref[-1])

    lo = hi - 1
    interval = q_ref[hi] - q_ref[lo]
    if interval == 0:  # duplicate q values, nothing to interpolate over
        return float(e_ref[lo])

    weight = (q - q_ref[lo]) / interval
    return float(np.sqrt(((1.0 - weight) * e_ref[lo]) ** 2 + (weight * e_ref[hi]) ** 2))


def valid_point_mask(profile: IQData) -> np.ndarray:
    """Mark the valid points of a scaled profile.

    Zero-count points (zero intensity) carry no information on the scale of the profile and
    are not valid points. They are excluded from the Q range of the profile, from the count of
    points in an overlap, from the scale-factor fit, and from the stitched profile.

    Parameters
    ----------
    profile : IQData
        Scaled profile of one order.

    Returns
    -------
    np.ndarray
        Boolean array, True for the points with non-zero intensity.
    """
    return np.asarray(profile.i, dtype=float) != 0


def overlap_range(q_ref: np.ndarray, q_n: np.ndarray) -> tuple[float, float]:
    """Q-range intersection of two profiles.

    Parameters
    ----------
    q_ref : np.ndarray
        Momentum-transfer values of the reference profile, in ``1/angstrom``.
    q_n : np.ndarray
        Momentum-transfer values of the profile to be scaled, in ``1/angstrom``.

    Returns
    -------
    tuple[float, float]
        ``(q_min, q_max)`` of the intersection, in ``1/angstrom``.

    Raises
    ------
    StitchError
        If either profile is empty or the two Q ranges do not intersect.
    """
    if len(q_ref) == 0 or len(q_n) == 0:
        raise StitchError("one of the profiles has no points")
    q_min = max(float(np.min(q_ref)), float(np.min(q_n)))
    q_max = min(float(np.max(q_ref)), float(np.max(q_n)))
    if q_min > q_max:
        raise StitchError(
            f"the Q ranges [{np.min(q_ref):.6g}, {np.max(q_ref):.6g}] and "
            f"[{np.min(q_n):.6g}, {np.max(q_n):.6g}] 1/angstrom do not overlap"
        )
    return q_min, q_max


def interpolate_profile(
    q_target: np.ndarray, q: np.ndarray, i: np.ndarray, e: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a profile onto the momentum-transfer points ``q_target``.

    Intensities are interpolated linearly, and uncertainties in variance with squared weights
    (see ``interpolate_error``). Extrapolation is not allowed.

    Parameters
    ----------
    q_target : np.ndarray
        Momentum-transfer points to interpolate onto, in ``1/angstrom``.
    q : np.ndarray
        Momentum-transfer values of the profile, in ``1/angstrom``, sorted in ascending order.
    i : np.ndarray
        Intensities of the profile.
    e : np.ndarray
        Uncertainties of the profile.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Interpolated intensities and uncertainties, one per entry of ``q_target``.

    Raises
    ------
    StitchError
        If any point of ``q_target`` lies outside the Q range of the profile.
    """
    if np.any(q_target < q[0]) or np.any(q_target > q[-1]):
        raise StitchError(
            f"interpolation outside the Q range [{q[0]:.6g}, {q[-1]:.6g}] 1/angstrom of the profile being scaled"
        )
    i_interp = np.interp(q_target, q, i)
    e_interp = np.array([interpolate_error(qt, q, e) for qt in q_target])
    return i_interp, e_interp


def scale_factor(
    i_ref: np.ndarray,
    e_ref: np.ndarray,
    i_n: np.ndarray,
    e_n: np.ndarray,
    rtol: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    r"""Error-weighted least-squares factor that scales a profile onto the reference.

    All inputs are on the same momentum-transfer points: the reference points in the overlap,
    with the profile being scaled interpolated onto them. The scale factor is

    .. math::

        s = \frac{\sum_k I_{\mathrm{ref},k}\, I_{n,k} / \sigma_k^2}{\sum_k I_{n,k}^2 / \sigma_k^2},
        \qquad
        \sigma_k^2 = \sigma_{\mathrm{ref},k}^2 + s^2\, \sigma_{n,k}^2

    where :math:`\sigma_k^2` is the variance of the residual
    :math:`I_{\mathrm{ref},k} - s\, I_{n,k}`, treating the two profiles as independent. The
    variance of the scaled profile enters because higher orders are usually noisier than the
    reference; leaving it out would give the noisiest points too much weight.

    Since :math:`\sigma_k^2` depends on :math:`s`, the equation has the form :math:`s = f(s)`.
    It is first solved by fixed-point iteration starting from :math:`s = 1`, until the relative
    change of :math:`s` falls below ``rtol``. The iteration converges only where
    :math:`|f'(s)| < 1` at the solution; otherwise it oscillates around the solution without
    approaching it. If it has not converged after ``max_iterations`` iterations, the root of
    :math:`g(s) = f(s) - s` is found with Brent's method instead. Because :math:`f(s)` is a
    weighted mean of the ratios :math:`r_k = I_{\mathrm{ref},k} / I_{n,k}` with non-negative
    weights :math:`I_{n,k}^2 / \sigma_k^2`, it lies between :math:`\min_k r_k` and
    :math:`\max_k r_k` for every :math:`s`. Hence :math:`g(\min_k r_k) \ge 0 \ge g(\max_k r_k)`,
    and these two ratios bracket a root, which Brent's method is guaranteed to find.

    The result is the self-consistent solution of the equation above. It is
    close to, but not identical with, the minimizer of
    :math:`\sum_k (I_{\mathrm{ref},k} - s\, I_{n,k})^2 / \sigma_k^2`, whose derivative has an
    extra term from the dependence of :math:`\sigma_k^2` on :math:`s`.

    Parameters
    ----------
    i_ref : np.ndarray
        Reference intensities.
    e_ref : np.ndarray
        Reference uncertainties.
    i_n : np.ndarray
        Intensities of the profile to be scaled, interpolated onto the reference points.
    e_n : np.ndarray
        Uncertainties of the profile to be scaled, interpolated onto the reference points.
    rtol : float
        Relative tolerance on the scale factor, for the fixed-point iteration and for Brent's
        method.
    max_iterations : int
        Maximum number of fixed-point iterations before falling back to Brent's method.

    Returns
    -------
    float
        The scale factor, finite and positive.

    Raises
    ------
    StitchError
        If some point has zero uncertainty in both profiles, if the scale factor is zero,
        negative, or not finite, or if the iteration does not converge and no positive
        solution can be bracketed.
    """
    var_ref = np.asarray(e_ref, dtype=float) ** 2
    var_n = np.asarray(e_n, dtype=float) ** 2
    if np.any((var_ref <= 0) & (var_n <= 0)):
        raise StitchError("some points in the overlap have zero uncertainty in both profiles")

    i_ref = np.asarray(i_ref, dtype=float)
    i_n = np.asarray(i_n, dtype=float)

    def update(s: float) -> float:
        """Right-hand side f(s) of the equation s = f(s)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = 1.0 / (var_ref + s**2 * var_n)
            return float(np.sum(weight * i_ref * i_n) / np.sum(weight * i_n**2))

    s = 1.0
    for _ in range(max_iterations):
        s_new = update(s)
        if not np.isfinite(s_new) or s_new <= 0:
            raise StitchError(f"invalid scale factor {s_new}")
        converged = abs(s_new - s) <= rtol * abs(s_new)
        s = s_new
        if converged:
            return s

    # The iteration oscillates; bracket the root of g(s) = f(s) - s by the extreme ratios
    # I_ref / I_n, between which f(s) always lies, and solve with Brent's method
    nonzero = i_n != 0
    ratios = i_ref[nonzero] / i_n[nonzero]
    s_low, s_high = float(np.min(ratios)), float(np.max(ratios))
    if not s_low > 0:
        raise StitchError(
            f"scale factor did not converge in {max_iterations} iterations, and the intensity ratios "
            f"[{s_low:.6g}, {s_high:.6g}] do not bracket a positive solution"
        )
    try:
        s = float(brentq(lambda x: update(x) - x, s_low, s_high, xtol=rtol * s_low, rtol=rtol))
    except (ValueError, RuntimeError) as error:
        raise StitchError(f"scale factor did not converge: {error}") from error
    if not np.isfinite(s) or s <= 0:
        raise StitchError(f"invalid scale factor {s}")
    logger.info(
        f"Scale factor iteration did not converge in {max_iterations} iterations; "
        f"solved with Brent's method instead, s = {s:.6g}."
    )
    return s


def merge_profiles(reference: IQData, scaled: IQData, q_start: float) -> IQData:
    """Merge the reference profile with a profile already scaled onto it.

    Every point of the reference profile is kept. Of the scaled profile, only the points with
    ``Q >= q_start`` are added, where ``q_start`` is the lower end of the fit range: the points
    inside the overlap that were used to fit the scale factor, and the points beyond the highest
    Q of the reference profile. The scaled points below ``q_start`` lie in the peak of the
    unscattered beam and are left out. The result is sorted by Q; the sort is stable, so for
    equal Q the reference point comes first.

    Parameters
    ----------
    reference : IQData
        The reference (stitched so far) profile.
    scaled : IQData
        The profile being added, already multiplied by its scale factor.
    q_start : float
        Lowest Q of the scaled points that are added, in ``1/angstrom``.

    Returns
    -------
    IQData
        The merged profile.
    """
    q_ref, i_ref, e_ref = (np.asarray(v, dtype=float) for v in (reference.q, reference.i, reference.e))
    q_n, i_n, e_n = (np.asarray(v, dtype=float) for v in (scaled.q, scaled.i, scaled.e))
    keep_n = q_n >= q_start

    q = np.concatenate([q_ref, q_n[keep_n]])
    i = np.concatenate([i_ref, i_n[keep_n]])
    e = np.concatenate([e_ref, e_n[keep_n]])
    order = np.argsort(q, kind="stable")
    return IQData(q=q[order].tolist(), i=i[order].tolist(), e=e[order].tolist(), t=[])


def _as_arrays(profile: IQData, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q, I, and E of a profile as arrays, checking that they are consistent.

    Parameters
    ----------
    profile : IQData
        Profile of one order.
    order : int
        Order (1-based) of the profile, used in error messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Momentum transfer in ``1/angstrom``, intensities, and uncertainties.

    Raises
    ------
    StitchError
        If the three arrays differ in length, or if Q is not sorted in ascending order.
    """
    q, i, e = (np.asarray(v, dtype=float) for v in (profile.q, profile.i, profile.e))
    if not len(q) == len(i) == len(e):
        raise StitchError(f"order {order} has Q, I, and E arrays of different lengths")
    if np.any(np.diff(q) < 0):
        raise StitchError(f"the Q values of order {order} are not sorted in ascending order")
    return q, i, e


def _valid_profile(profile: IQData, order: int) -> tuple[IQData, np.ndarray]:
    """Drop the zero-count points of a profile, requiring at least two valid points.

    Parameters
    ----------
    profile : IQData
        Scaled profile of one order.
    order : int
        Order (1-based) of the profile, used in error messages.

    Returns
    -------
    tuple[IQData, np.ndarray]
        The profile restricted to its valid points, and the mask of valid points over the
        input profile (see ``valid_point_mask``).

    Raises
    ------
    StitchError
        If the profile is inconsistent (see ``_as_arrays``) or has fewer than two valid points.
    """
    q, i, e = _as_arrays(profile, order)
    mask = valid_point_mask(profile)
    if np.count_nonzero(mask) < 2:
        raise StitchError(f"order {order} has fewer than two valid (non-zero-count) points")
    return IQData(q=q[mask].tolist(), i=i[mask].tolist(), e=e[mask].tolist(), t=[]), mask


def _join(
    current: IQData, profile: IQData, order: int, q_fit_min: float
) -> tuple[IQData, float, tuple[float, float], tuple[float, float], np.ndarray]:
    """Scale one order onto the current stitched profile and merge the two.

    Parameters
    ----------
    current : IQData
        Profile stitched so far, sorted by Q, without zero-count points.
    profile : IQData
        Scaled profile of the order being joined, sorted by Q.
    order : int
        Order (1-based) of ``profile``, used in log and error messages.
    q_fit_min : float
        Lower bound of the fit range, in ``1/angstrom``, excluding the unscattered-beam peak.

    Returns
    -------
    tuple[IQData, float, tuple[float, float], tuple[float, float], np.ndarray]
        The merged profile, the scale factor applied to ``profile``, the overlap
        ``(q_min, q_max)`` and the fit range ``(max(q_min, q_fit_min), q_max)`` in
        ``1/angstrom``, and the mask of valid points of ``profile``.

    Raises
    ------
    StitchError
        If ``profile`` has fewer than two valid points, the profiles do not overlap, the fit range
        holds fewer than two points of ``current``, interpolation would extrapolate, or the scale
        factor is invalid.
    QRangeNotExtendedError
        If the highest valid Q of ``profile`` does not exceed the highest Q of ``current``.
    """
    valid, mask = _valid_profile(profile, order)
    q_n, i_n, e_n = (np.asarray(v) for v in (valid.q, valid.i, valid.e))
    q_ref, i_ref, e_ref = (np.asarray(v) for v in (current.q, current.i, current.e))

    q_min, q_max = overlap_range(q_ref, q_n)
    # A weaker order that ends at or below the stitched profile adds no new Q range, only points
    # that are noisier than those already stitched
    if q_n[-1] <= q_ref[-1]:
        raise QRangeNotExtendedError(
            f"order {order} does not extend the stitched profile to higher Q (its highest "
            f"non-zero-count point is at {q_n[-1]:.6g} 1/angstrom, while the stitched profile "
            f"reaches {q_ref[-1]:.6g} 1/angstrom)"
        )
    q_start = max(q_min, q_fit_min)
    in_fit_range = (q_ref >= q_start) & (q_ref <= q_max)
    num_points = int(np.count_nonzero(in_fit_range))
    if num_points < 2:
        raise StitchError(
            f"the fit range [{q_start:.6g}, {q_max:.6g}] 1/angstrom of the overlap contains {num_points} "
            "point(s) of the stitched profile; at least two are required"
        )

    i_interp, e_interp = interpolate_profile(q_ref[in_fit_range], q_n, i_n, e_n)
    s = scale_factor(i_ref[in_fit_range], e_ref[in_fit_range], i_interp, e_interp)

    scaled = IQData(q=q_n.tolist(), i=(s * i_n).tolist(), e=(s * e_n).tolist(), t=[])
    logger.info(
        f"Joined order {order}: scale factor {s:.6g}, overlap [{q_min:.6g}, {q_max:.6g}] 1/angstrom, "
        f"fitted over [{q_start:.6g}, {q_max:.6g}] 1/angstrom with {num_points} points."
    )
    return merge_profiles(current, scaled, q_start), s, (q_min, q_max), (q_start, q_max), mask


def _stop_message(label: str, order: int, err: StitchError, joined_orders: list[int], num_orders: int) -> str:
    """Compose the log message for a failed join, which stops the stitching.

    Parameters
    ----------
    label : str
        Description of the measurement.
    order : int
        Order (1-based) whose join failed.
    err : StitchError
        Reason for the failure.
    joined_orders : list[int]
        Orders already in the stitched profile.
    num_orders : int
        Total number of orders given for stitching.

    Returns
    -------
    str
        The message, naming the orders in the stitched profile and those left out.
    """
    left_out = [n for n in range(1, num_orders + 1) if n not in joined_orders]
    return (
        f"Cannot join order {order} onto the stitched profile of {label}: {err}. "
        f"Stitching stops; the stitched profile contains orders {joined_orders}, "
        f"and orders {left_out} are left out."
    )


def stitch_profiles(profiles: list[IQData], label: str = "profile", q_fit_min: float = 0.0) -> StitchResult:
    """Stitch the scaled profiles of orders 1, 2, ... into a single profile.

    Order 1 is the fixed reference. Each higher order is joined onto the profile stitched so
    far:

    1. Zero-count points of the order are dropped (see ``valid_point_mask``).
    2. The overlap ``[q_min, q_max]`` is the Q-range intersection of the stitched profile and
       the order. The fit range is the part of it above the unscattered-beam peak,
       ``[max(q_min, q_fit_min), q_max]``.
    3. The order is interpolated onto the points of the stitched profile in the fit range, and
       scaled by the factor from ``scale_factor``. Uncertainties are scaled by the same factor.
    4. The two are merged with ``merge_profiles``: all points of the stitched profile are kept,
       and the points of the order from the lower end of the fit range upward are added.

    A join fails when the order has fewer than two valid points, when the profiles do not
    overlap, when the order does not extend the stitched profile to higher Q, when the fit range
    holds fewer than two points of the stitched profile, when interpolation would extrapolate,
    or when the scale factor is zero, negative, or not finite. Stitching then stops, and the
    result holds the orders joined up to that point.

    Each failure is logged as an error, except an order above 2 that does not extend the
    stitched profile to higher Q: higher orders are often too weak to carry counts at high Q,
    so that case is logged as a warning. For order 2 it is still an error, since no stitched
    profile can be produced without order 2.

    Parameters
    ----------
    profiles : list[IQData]
        Scaled profiles, entry ``n - 1`` holding order ``n``. Each must be sorted by Q, in
        ``1/angstrom``. A missing order can be given as an empty ``IQData``.
    label : str
        Description of the measurement, used in log messages.
    q_fit_min : float
        Lower bound of the fit range, in ``1/angstrom``, chosen above the peak of the unscattered
        beam (see ``usansred.reduce.unscattered_beam_q_cutoff``). The default, 0, fits over the
        whole overlap.

    Returns
    -------
    StitchResult
        The stitched profile together with the joined orders, their scale factors, the
        overlaps, the fit ranges, and the valid-point masks needed by ``apply_stitch``.

    Raises
    ------
    ValueError
        If ``profiles`` is empty.
    """
    if not profiles:
        raise ValueError("No profiles to stitch.")

    try:
        current, mask = _valid_profile(profiles[0], order=1)
    except StitchError as err:
        logger.error(f"Cannot stitch the orders of {label}: {err}.")
        return StitchResult()

    result = StitchResult(profile=current, joined_orders=[1], scale_factors=[1.0], masks=[mask])

    for order, profile in enumerate(profiles[1:], start=2):
        try:
            current, s, overlap, fit_range, mask = _join(current, profile, order, q_fit_min)
        except StitchError as err:
            level = err.log_level if order > 2 else logging.ERROR
            logger.log(level, _stop_message(label, order, err, result.joined_orders, len(profiles)))
            break
        result.profile = current
        result.joined_orders.append(order)
        result.scale_factors.append(s)
        result.overlaps.append(overlap)
        result.fit_ranges.append(fit_range)
        result.masks.append(mask)

    logger.info(
        f"Stitched orders {result.joined_orders} of {label} with scale factors "
        f"{[float(f'{s:.6g}') for s in result.scale_factors]}."
    )
    return result


def apply_stitch(profiles: list[IQData], result: StitchResult, label: str = "profile") -> StitchResult:
    """Stitch other profiles with the scale factors, masks, and fit ranges of an earlier stitch.

    Used for the background-subtracted profiles: the scale factors are computed once, from the
    scaled profiles, and reused here. Each profile must be on the same Q points as the scaled
    profile of the same order, which holds because background subtraction keeps the sample's
    Q points. The zero-count mask of the scaled profile is applied, since a background-subtracted
    intensity can be zero or negative without being a zero-count point.

    The stitch stops at the first order whose profile does not match its mask. An order with no
    points is logged as a warning: it is the expected outcome when the background has no data for
    that order, and ``Sample.subtract_background`` leaves an empty placeholder. Any other mismatch
    is an inconsistency and is logged as an error.

    Parameters
    ----------
    profiles : list[IQData]
        Profiles to stitch, entry ``n - 1`` holding order ``n``.
    result : StitchResult
        Outcome of ``stitch_profiles`` on the corresponding scaled profiles.
    label : str
        Description of the measurement, used in log messages.

    Returns
    -------
    StitchResult
        The stitched profile with the orders, scale factors, overlaps, fit ranges, and masks of ``result``
        that were used. These are the orders of ``result.joined_orders`` up to, and not
        including, the first one whose profile does not match its mask. The profile is empty
        when ``result`` holds no orders.
    """
    applied = StitchResult()
    for k, (order, s, mask) in enumerate(zip(result.joined_orders, result.scale_factors, result.masks)):
        profile = profiles[order - 1] if order <= len(profiles) else IQData()
        try:
            q, i, e = _as_arrays(profile, order)
            if len(q) == 0:
                raise MissingOrderError(f"order {order} has no data")
            if len(q) != len(mask):
                raise StitchError(f"order {order} has {len(q)} points, but its scaled profile has {len(mask)} points")
        except StitchError as err:
            logger.log(err.log_level, _stop_message(label, order, err, applied.joined_orders, len(profiles)))
            break
        scaled = IQData(q=q[mask].tolist(), i=(s * i[mask]).tolist(), e=(s * e[mask]).tolist(), t=[])
        if k == 0:
            applied.profile = scaled
        else:
            applied.profile = merge_profiles(applied.profile, scaled, result.fit_ranges[k - 1][0])
            applied.overlaps.append(result.overlaps[k - 1])
            applied.fit_ranges.append(result.fit_ranges[k - 1])
        applied.joined_orders.append(order)
        applied.scale_factors.append(s)
        applied.masks.append(mask)
    return applied
