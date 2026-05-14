"""Functions for finding a region of interest (ROI) in a transmission profile and integrating over it."""

import math

import numpy as np

from usansred.model import TransmissionData, XYData


def roi_from_profile(x: list[float], y: list[float], expand=0.25) -> tuple[float, float]:
    """Find center at max(Y), get half-max region, expand by 25% (clamped to array)"""
    if len(y) == 0:
        return (None, None)
    y = np.array(y)
    x = np.array(x)
    imax = int(np.argmax(y))
    ymax = y[imax]
    if ymax <= 0:
        return (None, None)
    mask = y >= 0.5 * ymax
    xs = x[mask]
    if xs.size == 0:
        return (None, None)
    xmin, xmax = xs.min(), xs.max()
    width = xmax - xmin
    xmin -= expand * width
    xmax += expand * width
    # clamp to data range
    xmin = max(xmin, x.min())
    xmax = min(xmax, x.max())
    return (xmin, xmax)


def integrate_over_roi(x: list[float], y: list[float], xmin: float, xmax: float) -> tuple[float, float]:
    """Simple rectangular integrationover points inside [xmin, xmax]"""
    x = np.array(x)
    y = np.array(y)
    m = (x >= xmin) & (x <= xmax)
    c = float(y[m].sum())
    # poisson-ish variance: sum(y)
    var = float(y[m].sum())
    return (c, var)


def get_raw_transmission_ratio(xy_trans: XYData, xy_mon: XYData) -> tuple[float, float]:
    """Calculate transmission ratio and error from transmission and monitor XYData.

    Arguments
    ---------
    xy_trans : XYData
        Transmission data
    xy_mon : XYData
        Monitor data

    Returns
    -------
    tuple
        Transmission ratio and error

    """
    if not (xy_trans.x and xy_trans.y and xy_mon.x and xy_mon.y):
        return (1.0, 0.0)

    xmin, xmax = roi_from_profile(xy_mon.x, xy_mon.y)
    if xmin is not None:
        c_trans, _ = integrate_over_roi(xy_trans.x, xy_trans.y, xmin, xmax)
        c_mon, _ = integrate_over_roi(xy_mon.x, xy_mon.y, xmin, xmax)
        if c_trans > 0 and c_mon > 0:
            raw_ratio = c_trans / c_mon
            # error of ratio: (Ct/Cm)^2 * (1/Ct + 1/Cm) under Poisson
            raw_ratio_err = raw_ratio * math.sqrt(1.0 / max(c_trans, 1.0) + 1.0 / max(c_mon, 1.0))

    return (raw_ratio, raw_ratio_err)


def get_bank_transmission(transmission: TransmissionData, trans_ref: int, trans_ref_err: int) -> tuple[float, float]:
    """Calculate per-bank transmission ratio and error from raw ratio and reference value.

    Arguments
    ---------
    transmission : TransmissionData
        Transmission data for the bank
    trans_ref : int
        Reference transmission value for the bank
    trans_ref_err : int
        Error in the reference transmission value for the bank

    Returns
    -------
    tuple
        Transmission ratio and error for the bank
    """
    raw_ratio = transmission.raw_ratio
    raw_E = transmission.raw_ratio_err
    ref = trans_ref if trans_ref > 0 else 1.0
    val = raw_ratio / ref

    # propagate error: val * sqrt( (raw_E/raw_ratio)^2 + (Eref/ref)^2 ), using Eref≈spread
    rel = 0.0
    if raw_ratio > 0:
        rel = (raw_E / raw_ratio) ** 2
    if trans_ref_err > 0 and ref > 0:
        rel += (trans_ref_err / ref) ** 2
    err = abs(val) * math.sqrt(rel) if rel > 0 else 0.0

    return (val, err)
