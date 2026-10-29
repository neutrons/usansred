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

Monitor Normalization
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

Analyzer Solid Angle
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

Sample Thickness
----------------

The scattered intensity is proportional to the thickness of the sample in the beam
(Section 4.2 of [NIST2008]_). The intensity is divided by the thickness :math:`t`, in cm, given
by the ``thickness`` key of the sample or background entry in the setup file.

The empty cell has no ``thickness`` key, because there is no sample in the beam (see
:doc:`measurement_types/empty_cell`).

.. _transmission-coefficient:

Transmission Coefficient
------------------------

The sample attenuates the beam through absorption and incoherent scattering, which reduces the
scattered intensity in proportion to the sample transmission :math:`T` (Eq. 1 in [NIST2008]_).
The transmission coefficient of a measurement is the fraction of the beam that passes through
it, relative to the empty cell. The intensities of every sample and of the background are
divided by their transmission coefficient.

The transmission coefficients are only computed when the setup file has an ``empty_cell`` entry
(see :doc:`measurement_types/empty_cell`). Without it, :math:`T = 1` for every measurement and
no transmission correction is applied. The empty cell itself is the reference and has
:math:`T = 1`.

Input Files
^^^^^^^^^^^

For every scan (run), the autoreduction writes three event-count files next to the rocking-curve
files. Each contains the counts of one detector summed over all its pixels, as a function of time
of flight in bins of 10 microseconds from 0 to 17000 microseconds:

- ``USANS_<run>_detector.txt``: the main detector, which receives the neutrons reflected by the
  analyzer.
- ``USANS_<run>_trans.txt``: the transmission detector, located behind the analyzer, which
  receives the neutrons not reflected by the analyzer.
- ``USANS_<run>_monitor.txt``: the beam monitor.

If one of these files is missing, the corresponding count is set to 0 and a warning is logged.

Calculation
^^^^^^^^^^^

For each measurement, the counts of the main detector :math:`N_D`, of the transmission detector
:math:`N_{Tr}`, and of the monitor :math:`N_M` are summed over all time-of-flight bins and over all
scans of the measurement, leaving out the scans listed in ``exclude``. The transmitted fraction of
the measurement is

.. math::

   \tau = \frac{N_D + N_{Tr}}{N_M}

and the transmission coefficient of a sample or of the background is the ratio of its transmitted
fraction to that of the empty cell:

.. math::

   T = \frac{\tau}{\tau_{empty \, cell}}

Why Both Detectors Are Counted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neutrons leaving the sample reach the analyzer, which reflects those meeting its Bragg condition
into the main detector and lets all others pass into the transmission detector. The split between
the two detectors therefore depends on the analyzer angle, and on how strongly the sample
scatters at small angles. At the NIST instrument, the transmission is measured with the analyzer
rotated far from the direct beam, so that no neutrons are reflected and the transmission detector
alone receives all of them ("wide-angle transmission", Section 5.4.1 of [NIST2008]_).

At the SNS USANS, the transmission is computed from the same scans used to measure the rocking
curves, during which the analyzer moves through many angles, including those near :math:`Q = 0`.
The count of either detector alone then depends on the analyzer angles of the scan. Their sum
includes the neutrons that passed through the sample whether the analyzer reflected them or not,
so it is much less sensitive to the analyzer angle. The resulting coefficient is the scan
equivalent of the wide-angle transmission.

Example
^^^^^^^

The table below compares, scan by scan, the transmission of sample ``S115_dry`` (runs 41712 to
41715) relative to ``EmptyPCell`` (runs 41740 to 41743), from the reference dataset
``IPTS-30410``. [#]_ The first two scans cross the direct beam, so a large share of the neutrons
is reflected into the main detector. The last two scans are at wide angles, where the main
detector receives almost no neutrons.

.. list-table::
   :header-rows: 1
   :widths: 10 18 18 18 18 18

   * - Scan
     - Share in main detector, empty cell
     - Share in main detector, sample
     - :math:`\tau_{empty \, cell}`
     - :math:`T` from both detectors
     - :math:`T` from transmission detector only
   * - 1
     - 0.289
     - 0.240
     - 0.3168
     - 0.990
     - 1.058
   * - 2
     - 0.390
     - 0.312
     - 0.3156
     - 1.003
     - 1.131
   * - 3
     - 0.001
     - 0.019
     - 0.3351
     - 0.984
     - 0.966
   * - 4
     - 0.000
     - 0.004
     - 0.3391
     - 0.985
     - 0.981

Using both detectors, the coefficient stays between 0.98 and 1.00 in every scan. Using the
transmission detector alone, it ranges from 0.97 to 1.13, with values above 1 in the scans that
cross the direct beam: the sample scatters neutrons out of the narrow acceptance of the analyzer,
so fewer reach the main detector and more reach the transmission detector. Summed over the four
scans, as ``usansred`` does, the transmission coefficient of ``S115_dry`` is :math:`T = 0.988`.

.. [#] The setup file of this dataset lists ``EmptyPCell`` as the background, so its reference
   reduction applies :math:`T = 1`. The example treats ``EmptyPCell`` as an empty cell for
   illustration.

Assumptions and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Detector response.** The counts of the main detector and of the transmission detector are
  added with equal weights. In the example above, the transmitted fraction of the empty cell is
  about 6% lower in the scans that cross the direct beam (0.316) than in the scans at wide angles
  (0.335 to 0.339). Neutrons reflected toward the main detector are therefore recorded with a
  lower overall efficiency than those reaching the transmission detector, either because of the
  detector efficiencies or because of losses in the analyzer. This effect largely cancels in
  :math:`T` as long as the sample and the empty cell are measured with the same sequence of scans.
- **Small-angle scattering.** The transmission detector is assumed to receive the neutrons
  scattered at small angles by the sample, so that only absorption and incoherent scattering
  reduce :math:`T`.
- **Wavelength.** The counts are summed over all times of flight, so :math:`T` is an average over
  wavelength. The same coefficient is applied to all harmonics and to all values of :math:`Q`.
- **Uncertainty.** The transmission coefficient is treated as exact. Its uncertainty is not
  included in the uncertainties of the reduced data.

Special Cases
^^^^^^^^^^^^^

- **No empty cell.** :math:`T = 1` for every measurement, and a message stating that the
  transmission correction is skipped is logged.
- **Empty cell with zero transmitted counts.** :math:`T` cannot be computed. :math:`T = 1` is used
  and a warning is logged.
- **Invalid coefficient.** If :math:`T` is zero, negative, or not a finite number, for example
  because a measurement has zero transmitted counts, the reduction stops with an error asking to
  check the transmitted counts of the measurement and of the empty cell.

Where to Find the Transmission Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transmission coefficient of each sample and of the background is written to its log file,
``reduction_<name>.log`` in the output folder, as::

   Transmission coefficient for sample S115_dry: 0.9881

It is not included in ``summary.xlsx`` or in the reduced data files.

Rocking Transmission
^^^^^^^^^^^^^^^^^^^^

The transmission coefficient computed by ``usansred`` is the equivalent of the wide-angle
transmission :math:`T_{wide}`, which accounts for absorption and incoherent scattering only.

NIST also describes a second transmission, the *rocking-curve transmission* :math:`T_{rock}`
(Section 5.4.2 of [NIST2008]_). The rocking curve of the direct beam is measured twice, with and
without the sample, by rotating the analyzer through :math:`Q = 0`. :math:`T_{rock}` is the ratio
of the two peak intensities at :math:`Q = 0`. Because neutrons scattered at small angles fall
outside the narrow acceptance of the analyzer, they are missing from the peak, so
:math:`T_{rock}` also accounts for the attenuation by coherent small-angle scattering.

The ratio of the two transmissions gives the scattering power :math:`\Sigma_{SAS} \cdot t` of the
sample, where :math:`\Sigma_{SAS}` is its macroscopic small-angle scattering cross section and
:math:`t` its thickness, and therefore an estimate of the amount of multiple scattering
(Eq. 10 and Section 5.5 of [NIST2008]_):

.. math::

   T_{SAS} = \frac{T_{rock}}{T_{wide}} = e^{-\Sigma_{SAS} \cdot t}

Multiple scattering is small when :math:`T_{SAS} > 0.9`.

``usansred`` does not currently calculate the rocking transmission :math:`T_{rock}`, nor the ratio
:math:`T_{SAS}`.

References
----------

.. [Agamalian2018] M. Agamalian, L. Heroux, K. C. Littrell, and J. M. Carpenter,
   "Progress on the Time-of-Flight Ultra Small Angle Neutron Scattering Instrument at SNS",
   *J. Phys.: Conf. Ser.* **1021**, 012033 (2018). https://doi.org/10.1088/1742-6596/1021/1/012033

.. [NIST2008] "SANS and USANS Investigation of the Structure of Coarse Fibrin Clots",
   Summer School on Small Angle Neutron Scattering and Neutron Reflectometry,
   NIST Center for Neutron Research, June 23-27, 2008.
