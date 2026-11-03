"""Unit tests for usansred.stitch, using synthetic profiles with known scale factors."""

import logging

import numpy as np
import pytest

from usansred.models import IQData
from usansred.stitch import (
    StitchError,
    apply_stitch,
    interpolate_error,
    interpolate_profile,
    merge_profiles,
    overlap_range,
    scale_factor,
    stitch_profiles,
    valid_point_mask,
)


def linear_intensity(q: np.ndarray) -> np.ndarray:
    """Synthetic intensity, linear in Q.

    Linear interpolation of a linear function is exact, so the stitching recovers the
    scale factors of synthetic profiles exactly.

    Parameters
    ----------
    q : np.ndarray
        Momentum transfer in ``1/angstrom``, below ``5e-3`` so that the intensity stays positive.

    Returns
    -------
    np.ndarray
        Intensity at each ``q``.
    """
    return 10.0 - 2000.0 * q


def make_profile(q: np.ndarray, scale: float = 1.0, relative_error: float = 0.05) -> IQData:
    """Synthetic profile ``linear_intensity(q) / scale``.

    Parameters
    ----------
    q : np.ndarray
        Momentum transfer in ``1/angstrom``, sorted in ascending order.
    scale : float
        Factor that the stitching must apply to this profile to match order 1.
    relative_error : float
        Uncertainty of each point, as a fraction of its absolute intensity.

    Returns
    -------
    IQData
        The synthetic profile.
    """
    i = linear_intensity(q) / scale
    return IQData(q=q.tolist(), i=i.tolist(), e=(relative_error * np.abs(i)).tolist(), t=[])


# Order n spans a Q range n times wider than order 1, on its own Q grid
Q1 = np.linspace(1.0e-5, 1.0e-3, 40)
Q2 = np.linspace(2.0e-5, 2.0e-3, 37)
Q3 = np.linspace(3.0e-5, 3.0e-3, 31)


class TestInterpolateError:
    """Uncertainty interpolation, propagated in variance with squared weights."""

    def test_midpoint_propagates_in_quadrature(self):
        """At the midpoint, both weights are 0.5, so sigma = sqrt((0.5 sigma_0)^2 + (0.5 sigma_1)^2)."""
        q_ref = np.array([1.0, 2.0])
        e_ref = np.array([3.0, 4.0])

        error = interpolate_error(1.5, q_ref, e_ref)

        assert error == pytest.approx(np.sqrt((0.5 * 3.0) ** 2 + (0.5 * 4.0) ** 2))

    @pytest.mark.parametrize(("q", "expected"), [(0.5, 3.0), (2.5, 4.0)])
    def test_outside_range_returns_nearest_endpoint(self, q, expected):
        """Outside the Q range, the nearest endpoint uncertainty is returned, as numpy.interp does."""
        assert interpolate_error(q, np.array([1.0, 2.0]), np.array([3.0, 4.0])) == expected


class TestHelpers:
    """Building blocks of a join: valid points, overlap, interpolation, and merge."""

    def test_valid_point_mask_marks_zero_counts(self):
        """Only zero intensity marks a zero-count point; negative intensities remain valid."""
        profile = IQData(q=[1.0, 2.0, 3.0], i=[1.0, 0.0, -1.0], e=[0.1, 0.1, 0.1])

        assert valid_point_mask(profile).tolist() == [True, False, True]

    def test_overlap_range_is_intersection(self):
        """The overlap of orders 1 and 2 runs from the lowest Q of order 2 to the highest Q of order 1."""
        assert overlap_range(Q1, Q2) == (Q2[0], Q1[-1])

    def test_overlap_range_disjoint_raises(self):
        """Profiles with disjoint Q ranges cannot be joined."""
        with pytest.raises(StitchError, match="do not overlap"):
            overlap_range(np.array([1.0, 2.0]), np.array([3.0, 4.0]))

    def test_interpolate_profile_rejects_extrapolation(self):
        """Interpolating onto a Q point outside the range of the profile raises instead of extrapolating."""
        with pytest.raises(StitchError, match="interpolation outside"):
            interpolate_profile(np.array([0.5, 1.5]), np.array([1.0, 2.0]), np.ones(2), np.ones(2))

    def test_merge_keeps_both_sets_inside_overlap_and_sorts(self):
        """The merge keeps reference points with Q < q_max and scaled points with Q > q_min, sorted by Q.

        Inside the overlap [2.5, 4.0], points of both profiles are kept. The reference point at
        q_max = 4.0 and the scaled point at q_min = 2.5 are dropped; the scaled point at 4.0 is kept.
        """
        reference = IQData(q=[1.0, 2.0, 3.0, 4.0], i=[1.0] * 4, e=[0.1] * 4)
        scaled = IQData(q=[2.5, 3.5, 4.0, 5.0], i=[2.0] * 4, e=[0.2] * 4)

        merged = merge_profiles(reference, scaled, q_min=2.5, q_max=4.0)

        # Reference points with Q < 4.0 and scaled points with Q > 2.5
        assert merged.q == [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
        assert merged.i == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


class TestScaleFactor:
    """Error-weighted least-squares scale factor and its validation."""

    def test_exact_data_recovers_scale(self):
        """When the scaled profile is exactly the reference divided by s, the fit returns s."""
        i_ref = np.array([4.0, 6.0, 8.0])
        i_n = i_ref / 2.5

        s = scale_factor(i_ref, 0.1 * i_ref, i_n, 0.1 * i_n)

        assert s == pytest.approx(2.5, rel=1e-12)

    def test_solution_is_self_consistent_with_full_variance(self):
        """The scale factor solves s = sum(I_ref I_n w) / sum(I_n^2 w), with w = 1 / (var_ref + s^2 var_n).

        With noisy data, the result also differs from the fit that weights by the reference variance
        only, showing that the variance of the scaled profile enters the weights.
        """
        rng = np.random.default_rng(42)
        i_ref = np.linspace(10.0, 2.0, 20)
        e_ref = 0.05 * i_ref
        i_n = i_ref / 3.0 * (1 + 0.1 * rng.standard_normal(20))
        e_n = 0.2 * np.abs(i_n)

        s = scale_factor(i_ref, e_ref, i_n, e_n)

        weight = 1.0 / (e_ref**2 + s**2 * e_n**2)
        assert s == pytest.approx(np.sum(weight * i_ref * i_n) / np.sum(weight * i_n**2), rel=1e-9)
        # The variance of the scaled profile changes the result compared to reference-only weights
        weight_ref = 1.0 / e_ref**2
        s_ref_only = np.sum(weight_ref * i_ref * i_n) / np.sum(weight_ref * i_n**2)
        assert s != pytest.approx(s_ref_only, rel=1e-6)

    def test_negative_scale_raises(self):
        """A profile of opposite sign to the reference gives a negative scale factor, which is invalid."""
        i_ref = np.array([4.0, 6.0])

        with pytest.raises(StitchError, match="invalid scale factor"):
            scale_factor(i_ref, 0.1 * i_ref, -i_ref, 0.1 * i_ref)

    def test_non_finite_scale_raises(self):
        """A profile of zero intensity in the overlap gives a zero denominator and a non-finite scale factor."""
        i_ref = np.array([4.0, 6.0])

        with pytest.raises(StitchError, match="invalid scale factor"):
            scale_factor(i_ref, 0.1 * i_ref, np.zeros(2), np.full(2, 0.1))

    def test_zero_uncertainties_raise(self):
        """Points with zero uncertainty in both profiles have an undefined weight, so the fit is rejected."""
        i_ref = np.array([4.0, 6.0])

        with pytest.raises(StitchError, match="zero uncertainty"):
            scale_factor(i_ref, np.zeros(2), i_ref, np.zeros(2))


class TestStitchProfiles:
    """Sequential stitching of orders 1, 2, ... onto the profile stitched so far."""

    def test_two_orders_on_different_grids(self):
        """Order 2, on its own Q grid, is scaled onto order 1 with the known scale factor.

        The stitched profile is sorted by Q, keeps order-1 points below the overlap end and order-2
        points above the overlap start, and lies on the order-1 curve.
        """
        profiles = [make_profile(Q1), make_profile(Q2, scale=2.5)]

        result = stitch_profiles(profiles)

        assert result.joined_orders == [1, 2]
        assert result.scale_factors[0] == 1.0
        assert result.scale_factors[1] == pytest.approx(2.5, rel=1e-9)
        assert result.overlaps == [(Q2[0], Q1[-1])]
        q = np.array(result.profile.q)
        assert np.all(np.diff(q) >= 0)
        # Order-1 points with Q < Q1[-1] and order-2 points with Q > Q2[0]
        assert len(q) == (len(Q1) - 1) + (len(Q2) - 1)
        # Every stitched point lies on the order-1 curve
        np.testing.assert_allclose(result.profile.i, linear_intensity(q), rtol=1e-9)

    def test_scaled_errors_are_multiplied_by_scale(self):
        """Uncertainties are multiplied by the scale factor, which preserves the relative uncertainty.

        The stitched profile also extends to the highest Q of order 2.
        """
        profiles = [make_profile(Q1), make_profile(Q2, scale=2.5)]

        result = stitch_profiles(profiles)

        q = np.array(result.profile.q)
        i = np.array(result.profile.i)
        # make_profile uses 5 % relative errors, which scaling preserves
        np.testing.assert_allclose(result.profile.e, 0.05 * np.abs(i), rtol=1e-9)
        assert q[-1] == Q2[-1]

    def test_three_orders_scaled_onto_running_profile(self):
        """Order 3 is scaled onto the stitched orders 1 and 2, not onto order 1 or order 2 alone.

        Its overlap runs from the lowest Q of order 3 to the highest Q of order 2, and all scale
        factors are relative to order 1.
        """
        profiles = [make_profile(Q1), make_profile(Q2, scale=2.5), make_profile(Q3, scale=0.4)]

        result = stitch_profiles(profiles)

        assert result.joined_orders == [1, 2, 3]
        np.testing.assert_allclose(result.scale_factors, [1.0, 2.5, 0.4], rtol=1e-9)
        assert result.overlaps[1] == (Q3[0], Q2[-1])
        np.testing.assert_allclose(result.profile.i, linear_intensity(np.array(result.profile.q)), rtol=1e-9)

    def test_zero_count_points_excluded(self):
        """Zero-count points of order 2 are left out of the fit, the masks, and the stitched profile.

        A zero-count point at the upper end of order 2 does not extend the stitched profile, which
        ends at the last valid point of order 2.
        """
        order2 = make_profile(Q2, scale=2.5)
        # Zero-count points inside the overlap and at the upper end of order 2
        for index in (3, 10, len(Q2) - 1):
            order2.i[index] = 0.0
        profiles = [make_profile(Q1), order2]

        result = stitch_profiles(profiles)

        # Zero-count points would bias the fit; they are ignored, so s is exact
        assert result.scale_factors[1] == pytest.approx(2.5, rel=1e-9)
        assert 0.0 not in result.profile.i
        assert result.masks[1].sum() == len(Q2) - 3
        # The upper end of the stitched profile is the last valid point of order 2
        assert result.profile.q[-1] == Q2[-2]

    def test_no_overlap_stops_stitching(self, caplog):
        """An order whose Q range does not overlap the stitched profile is not joined.

        An error naming the measurement is logged, and the result holds order 1 only.
        """
        profiles = [make_profile(Q1), make_profile(Q1 + 1.0e-2, scale=2.5)]

        with caplog.at_level(logging.ERROR):
            result = stitch_profiles(profiles, label="sample S1")

        assert result.joined_orders == [1]
        assert result.profile.q == Q1.tolist()
        assert "Cannot join order 2 onto the stitched profile of sample S1" in caplog.text
        assert "do not overlap" in caplog.text

    def test_fewer_than_two_points_in_overlap(self, caplog):
        """A join needs at least two points of the stitched profile in the overlap.

        Order 2 starts at the highest Q of order 1, so the overlap holds a single point.
        """
        # Order 2 starts at the last point of order 1, so the overlap holds a single point
        profiles = [make_profile(Q1), make_profile(np.linspace(Q1[-1], 2.0e-3, 10), scale=2.5)]

        with caplog.at_level(logging.ERROR):
            result = stitch_profiles(profiles)

        assert result.joined_orders == [1]
        assert "at least two are required" in caplog.text

    def test_order_with_fewer_than_two_valid_points(self, caplog):
        """An order with a single non-zero-count point cannot be joined."""
        order2 = make_profile(Q2, scale=2.5)
        order2.i = [0.0] * (len(Q2) - 1) + [order2.i[-1]]

        with caplog.at_level(logging.ERROR):
            result = stitch_profiles([make_profile(Q1), order2])

        assert result.joined_orders == [1]
        assert "fewer than two valid" in caplog.text

    def test_invalid_scale_factor(self, caplog):
        """An order with negative intensities gives a negative scale factor; the join fails with an error."""
        order2 = make_profile(Q2, scale=-2.5)  # negative intensities

        with caplog.at_level(logging.ERROR):
            result = stitch_profiles([make_profile(Q1), order2])

        assert result.joined_orders == [1]
        assert "invalid scale factor" in caplog.text

    def test_stops_at_first_failed_join(self, caplog):
        """Stitching stops at the first failed join, here a missing order 2.

        Order 3 could be joined onto order 1, but it is not attempted.
        """
        profiles = [make_profile(Q1), IQData(), make_profile(Q3, scale=0.4)]

        with caplog.at_level(logging.ERROR):
            result = stitch_profiles(profiles)

        assert result.joined_orders == [1]
        assert result.scale_factors == [1.0]
        assert "Cannot join order 2" in caplog.text
        assert "the stitched profile contains orders [1], and orders [2, 3] are left out" in caplog.text
        assert "Cannot join order 3" not in caplog.text

    def test_order_not_extending_q_range_stops_with_warning(self, caplog):
        """An order above 2 that does not reach beyond the stitched profile stops the stitching.

        Order 3 has zero counts above 1.5e-3 1/angstrom, below the highest Q of order 2
        (2.0e-3 1/angstrom). Merging it would discard the order-2 points above 1.5e-3, so the
        join is rejected. This is the usual outcome for weak higher orders, so it is logged as
        a warning, and orders 3 and 4 are named as left out.
        """
        order3 = make_profile(Q3, scale=0.4)
        order3.i = [i if q <= 1.5e-3 else 0.0 for q, i in zip(order3.q, order3.i)]
        profiles = [make_profile(Q1), make_profile(Q2, scale=2.5), order3, make_profile(4 * Q1, scale=0.1)]

        with caplog.at_level(logging.WARNING):
            result = stitch_profiles(profiles, label="sample S1")

        assert result.joined_orders == [1, 2]
        assert result.profile.q[-1] == Q2[-1]
        [record] = [r for r in caplog.records if "Cannot join" in r.getMessage()]
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        assert "Cannot join order 3 onto the stitched profile of sample S1" in message
        assert "order 3 does not extend the stitched profile to higher Q" in message
        assert f"reaches {Q2[-1]:.6g} 1/angstrom" in message
        assert "the stitched profile contains orders [1, 2], and orders [3, 4] are left out" in message

    def test_order_2_not_extending_q_range_is_an_error(self, caplog):
        """When order 2 does not reach beyond order 1, the failure is logged as an error.

        Without order 2 no stitched profile can be produced, so this case is not downgraded
        to a warning like it is for higher orders.
        """
        order2 = make_profile(Q2, scale=2.5)
        order2.i = [i if q <= 8.0e-4 else 0.0 for q, i in zip(order2.q, order2.i)]

        with caplog.at_level(logging.WARNING):
            result = stitch_profiles([make_profile(Q1), order2])

        assert result.joined_orders == [1]
        [record] = [r for r in caplog.records if "Cannot join" in r.getMessage()]
        assert record.levelno == logging.ERROR
        assert "order 2 does not extend the stitched profile to higher Q" in record.getMessage()

    def test_invalid_first_order_returns_empty_result(self, caplog):
        """When order 1 has no valid points, nothing can be stitched and the result is empty."""
        with caplog.at_level(logging.ERROR):
            result = stitch_profiles([IQData(), make_profile(Q2)])

        assert result.joined_orders == []
        assert result.profile.q == []
        assert "Cannot stitch the orders" in caplog.text

    def test_empty_list_raises(self):
        """Calling stitch_profiles without profiles is a programming error."""
        with pytest.raises(ValueError, match="No profiles"):
            stitch_profiles([])


class TestApplyStitch:
    """Reuse of an earlier stitch for the background-subtracted profiles."""

    def test_reuses_scale_factors_masks_and_overlaps(self):
        """The background-subtracted profiles are stitched with the scale factors of the scaled profiles.

        A constant background b_n on order n becomes s_n * b_n after scaling. The backgrounds are
        chosen so that s_n * b_n = 1 for every order, so the stitched background-subtracted profile
        is the stitched scaled profile minus 1, on the same Q points and with the same uncertainties.
        """
        scaled = [make_profile(Q1), make_profile(Q2, scale=2.5), make_profile(Q3, scale=0.4)]
        scaled[1].i[5] = 0.0  # zero-count point
        result = stitch_profiles(scaled)
        # Background-subtracted profiles on the same Q points; a constant background per order
        backgrounds = [1.0, 0.4, 2.5]
        subtracted = [IQData(q=p.q, i=(np.array(p.i) - b).tolist(), e=p.e, t=[]) for p, b in zip(scaled, backgrounds)]

        applied = apply_stitch(subtracted, result)

        assert applied.joined_orders == result.joined_orders
        assert applied.scale_factors == result.scale_factors
        assert applied.overlaps == result.overlaps
        stitched = applied.profile
        assert stitched.q == result.profile.q
        # A background of b on order n becomes s_n * b after scaling, i.e. 1.0 for every order here
        np.testing.assert_allclose(np.array(result.profile.i) - np.array(stitched.i), 1.0, rtol=1e-9)
        np.testing.assert_allclose(stitched.e, result.profile.e)

    def test_mask_applies_even_where_subtracted_intensity_is_nonzero(self):
        """The zero-count mask comes from the scaled profiles.

        A point with zero scaled intensity is dropped even though its background-subtracted
        intensity is non-zero.
        """
        scaled = [make_profile(Q1), make_profile(Q2, scale=2.5)]
        scaled[1].i[5] = 0.0
        result = stitch_profiles(scaled)
        subtracted = [IQData(q=p.q, i=(np.array(p.i) - 1.0).tolist(), e=p.e, t=[]) for p in scaled]

        stitched = apply_stitch(subtracted, result).profile

        assert Q2[5] not in stitched.q
        assert len(stitched.q) == len(result.profile.q)

    def test_mismatched_profile_stops(self, caplog):
        """A background-subtracted order whose points do not match its mask stops the stitch with an error."""
        scaled = [make_profile(Q1), make_profile(Q2, scale=2.5)]
        result = stitch_profiles(scaled)
        subtracted = [scaled[0], IQData()]

        with caplog.at_level(logging.ERROR):
            applied = apply_stitch(subtracted, result, label="sample S1")

        assert applied.profile.q == Q1.tolist()
        assert applied.joined_orders == [1]
        assert "Cannot join order 2 onto the stitched profile of sample S1" in caplog.text

    def test_empty_result_gives_empty_profile(self):
        """When the scaled profiles could not be stitched, the background-subtracted result is empty."""
        applied = apply_stitch([make_profile(Q1)], stitch_profiles([IQData()]))

        assert applied.profile.q == []
        assert applied.joined_orders == []
