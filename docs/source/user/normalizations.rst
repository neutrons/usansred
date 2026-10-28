.. normalizations

Normalizations
==============

The intensity measured on the detector depends not only on the scattering from the sample, but
also on the number of neutrons incident on the sample, on the solid angle accepted by the
analyzer, on the sample thickness, and on the attenuation of the beam by the sample
(Eq. 1 in [NIST2008]_). ``usansred`` removes these dependencies with four normalizations:

- `Monitor normalization`_, applied to every scan before stitching.
- `Analyzer solid angle`_, `Sample thickness`_, and `Transmission coefficient`_, applied together
  to the stitched data of each harmonic :math:`n` after conversion to momentum transfer
  :math:`Q_n`:

  .. math::

     I_{scaled}(Q_n) = \frac{I(Q_n)}{\Delta\Omega_n \, t \, T}, \qquad
     \sigma_{scaled}(Q_n) = \frac{\sigma_I(Q_n)}{\Delta\Omega_n \, t \, T}

The place of these normalizations in the reduction is described in :doc:`reduction_workflow`.

.. _monitor-normalization:

Monitor normalization
---------------------

The number of neutrons reaching the sample varies during and between scans. To remove this
variation, the detector counts :math:`D` of every harmonic are divided by the monitor counts
:math:`M` collected while the analyzer remained at the same angle:

.. math::

   I = \frac{D}{M}, \qquad
   \sigma_I = \frac{\sqrt{\sigma_D^2 + \left(I \, \sigma_M\right)^2}}{M}

where the detector and monitor counts are treated as statistically independent.

Monitor normalization makes the intensities proportional to the scattering cross section of the
sample, but it does not put them on an absolute scale (see
:ref:`units of the reduced data <units-of-reduced-data>`).

Analyzer solid angle
--------------------

The analyzer accepts scattered neutrons over a very narrow range of horizontal angles, but over a
wide range of vertical angles ("slit geometry", Section 6 of [NIST2008]_). The solid angle accepted
by the analyzer for harmonic :math:`n` is the product of the two angular widths:

.. math::

   \Delta\Omega_n = \Delta\theta_v \, \Delta\theta_{h,n}

- :math:`\Delta\theta_v` is the full vertical angular acceptance, that is, the vertical divergence
  of the beam convoluted with the vertical angular range accepted by the detector (the quantity
  :math:`2\Delta q_v` of Eq. 12 in [NIST2008]_, expressed as an angle). ``usansred`` uses
  :math:`\Delta\theta_v = 0.042` rad for the SNS USANS instrument, for all harmonics.
- :math:`\Delta\theta_{h,n}` is the horizontal angular acceptance, given by the full width at half
  maximum (FWHM) of the rocking curve of harmonic :math:`n`. ``usansred`` uses

  .. math::

     \Delta\theta_{h,n} = \frac{5.34 \, e^{-0.01793 \, n^2}}{n^2} \ \text{arcsec}

  a fit to the rocking-curve widths measured for orders :math:`n = 1` to 6 (Table 1 in
  [Agamalian2018]_), converted to radians with :math:`1'' = \pi / 648000` rad.

:math:`\Delta\Omega_n` is in sr.

Sample thickness
----------------

The scattered intensity is proportional to the thickness of the sample in the beam
(Section 4.2 of [NIST2008]_). The intensity is divided by the thickness :math:`t`, in cm, given
by the ``thickness`` key of the sample or background entry in the setup file.

The empty cell has no ``thickness`` key, because there is no sample in the beam (see
:doc:`measurement_types/empty_cell`).

Transmission coefficient
------------------------

The sample attenuates the beam through absorption and scattering, which reduces the scattered
intensity in proportion to the sample transmission :math:`T` (Eq. 1 in [NIST2008]_). The
intensity is divided by the transmission coefficient of the measurement, computed relative to
the empty cell (see :doc:`measurement_types/empty_cell`). Without an empty cell, :math:`T = 1`.

The transmission coefficient is treated as exact: its uncertainty is not included in
:math:`\sigma_{scaled}`.

References
----------

.. [Agamalian2018] M. Agamalian, L. Heroux, K. C. Littrell, and J. M. Carpenter,
   "Progress on the Time-of-Flight Ultra Small Angle Neutron Scattering Instrument at SNS",
   *J. Phys.: Conf. Ser.* **1021**, 012033 (2018). https://doi.org/10.1088/1742-6596/1021/1/012033

.. [NIST2008] "SANS and USANS Investigation of the Structure of Coarse Fibrin Clots",
   Summer School on Small Angle Neutron Scattering and Neutron Reflectometry,
   NIST Center for Neutron Research, June 23-27, 2008.
