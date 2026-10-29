.. reduction_workflow

Reduction Workflow
==================

Each scan rotates the analyzer through a series of angles :math:`\theta`, measured in arcsec,
and records the detector and monitor counts at each angle. The detector counts are recorded
separately for each harmonic :math:`n = 1, \ldots, 4`, of wavelength :math:`\lambda_n = \lambda_1 / n`
with :math:`\lambda_1 = 3.6` angstrom (Table 1 in [Agamalian2018]_), so each scan provides one
rocking curve per harmonic.

Each measurement (sample, background, or empty cell, see :doc:`measurement_types`) is reduced (the
empty cell only when no background is present) in the following steps, applied to every harmonic:

1. `Counting uncertainties`_ are assigned to the detector and monitor counts of every scan.
2. The detector counts of every scan are divided by the monitor counts
   (:ref:`monitor normalization <monitor-normalization>`).
3. `Stitching`_ combines the scans into a single rocking curve.
4. `Centering`_ shifts the rocking curve so that the unscattered beam is at :math:`\theta = 0`.
5. `Conversion to momentum transfer`_ maps the centered angles to :math:`Q_n`.
6. The intensities are divided by the analyzer solid angle, the sample thickness, and the
   transmission coefficient (see :doc:`normalizations`).
7. `Points at the same momentum transfer`_ are averaged.
8. For samples, the background or the empty cell is subtracted (see :doc:`measurement_types`).

The units of the resulting output files are described in
:ref:`units of the reduced data <units-of-reduced-data>`.

Counting Uncertainties
----------------------

The uncertainty of a number of counts :math:`N` is computed as

.. math::

   \sigma_N = \sqrt{|N - 0.5| + 0.5}

which is close to the Poisson uncertainty :math:`\sqrt{N}` for large counts, and equal to 1 for
:math:`N = 0` and :math:`N = 1`, so that no data point has zero uncertainty. The uncertainty
column of the input files is not used.

Stitching
---------

The scans of a measurement cover different, partly overlapping ranges of analyzer angle. The
monitor-normalized points of all scans are merged into a single rocking curve sorted by angle.
When two or more scans contain the same analyzer angle, their intensities are combined with
inverse-variance weights:

.. math::

   I = \frac{\sum_i I_i / \sigma_i^2}{\sum_i 1 / \sigma_i^2}, \qquad
   \sigma_I = \left( \sum_i \frac{1}{\sigma_i^2} \right)^{-1/2}

Centering
---------

The analyzer angle at which the unscattered beam is reflected may differ slightly from the
nominal zero of the analyzer motor. This angle :math:`\theta_0` is found by fitting a Gaussian
peak on a constant background,

.. math::

   I(\theta) = b + A \exp\left[ -\frac{(\theta - \theta_0)^2}{2 s^2} \right]

to the rocking curve of the first harmonic, weighted by the uncertainties of the data. The fit is
restricted to the symmetric angle range :math:`[\theta_{min}, -\theta_{min}]`, where
:math:`\theta_{min} < 0` is the smallest analyzer angle measured. Only the fitted center
:math:`\theta_0` is used: it is subtracted from the angles of every harmonic, so that all
rocking curves of the measurement share the same zero.

Conversion to Momentum Transfer
-------------------------------

For the very small scattering angles measured on the USANS, the momentum transfer is
proportional to the centered analyzer angle:

.. math::

   Q_n = \frac{2\pi}{\lambda_n} \, |\theta - \theta_0|
       = \frac{2\pi n}{\lambda_1} \, |\theta - \theta_0|

with the angles converted from arcsec to radians (:math:`1'' = \pi / 648000` rad) and
:math:`Q_n` in ``1/angstrom``. Negative and positive angles of the same magnitude map to the same
:math:`Q_n`.

Points at the Same Momentum Transfer
------------------------------------

When a negative and a positive analyzer angle map to the same :math:`Q_n`, the :math:`N` points
at that :math:`Q_n` are averaged:

.. math::

   I = \frac{1}{N} \sum_i I_i, \qquad
   \sigma_I = \frac{1}{N} \sqrt{\sum_i \sigma_i^2}
