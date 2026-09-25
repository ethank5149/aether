"""Spectral radiation transport for nonequilibrium air plasmas.

HARA-style emission model with smeared rotational bands for molecular
systems (N2 first/second positive, NO beta/gamma) and Voigt-profiled
atomic lines (N I, O I).  Two-temperature Boltzmann populations
(T_tr for rotation and atomic excitation, T_ve for vibration and
molecular electronic excitation) consistent with Park's two-temperature
model used by SU2 NEMO.

Ray-marching radiative transfer through unstructured CFD volumes with
CIE 1931 spectral-to-sRGB conversion produces physically-grounded
composite images of the radiating shock layer.
"""
