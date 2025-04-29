#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for physically interpreting 
SSH / SST outputiles

1st Author: Tatsu
Date: First version: 04.24.2025

Dependencies:
    numpy
    scipy    
    xarray
    xrft

"""

import numpy as np
import scipy.signal as signal
import xarray as xr
import numpy as np
import requests
import scipy.interpolate
import traceback

import xrft




def compute_power_spectra_xrft(patch, field, dx, assert_parseval=False):
    """
    Compute power spectra for SWOT SSH data using xarray and xrft.

    Parameters
    ----------


    Returns
    -------
    
     """

    # Would be nice to convert SSH to centimeters if possible
    if field == "SSH":
        patch = patch * 100
    
    # Working under the assumption that we're not working with Nan values atm..
    #msk = np.isnan(patch)
    #swath = swath[:, ~msk]

    # Initialize arrays to store frequency and power spectral density for this patch
    freqs_x = np.zeros(patch.shape)
    psds_x = np.zeros(patch.shape)

    # Compute the FFT and power spectrum for each cross-swath row
    for i in range(len(patch)):
        patch_i = patch[i, :]  # Extract one row
        Nx = patch_i.size  # Number of points
        dx = dx  # Spacing between points
        da = xr.DataArray(patch_i, dims="x", coords={"x": dx * np.arange(0, patch_i.size)})

        # Compute the Discrete Fourier Transform
        FT = xrft.dft(da, dim="x", true_phase=True, true_amplitude=True)

        # Compute the power spectrum
        ps = xrft.power_spectrum(da, dim="x")

        # Store frequencies and power spectral densities
        freqs_x[i, :] = FT["freq_x"].values
        psds_x[i, :] = ps.values

        if assert_parseval:
            # Perform Parseval's theorem checks
            print("Parseval's theorem directly from FFT:")
            print(((np.abs(da) ** 2).sum() * dx).values)
            print(((np.abs(FT) ** 2).sum() * FT["freq_x"].spacing).values)
            print("Parseval's theorem from power spectrum:")
            print(ps.sum().values)
            print(((np.abs(da) ** 2).sum() * dx).values)
            print("Result:",((np.abs(da) ** 2).sum() * dx).values == ((np.abs(FT) ** 2).sum() * FT["freq_x"].spacing).values)
            print()
            

    # Return the computed 1D spectra
    return freqs_x, psds_x
