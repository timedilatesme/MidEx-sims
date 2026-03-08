"""
midex_instrument.py

Single Source of Truth for the MidEx simulated observatory.
Handles speclite filter registration, lenstronomy image observation kwargs,
and SkyPy configuration generation.
"""

import numpy as np
import os
import copy
from astropy import units as u
import speclite.filters
import lenstronomy.Util.util as util

# ==========================================
# 1. MASTER INSTRUMENT DICTIONARY FOR MidEx
# ==========================================
MIDEX_BANDS = {
    "A": {
        "wave_center": 4000.0, "wave_width": 1000.0,             # top-hat filter properties
        "exposure_time": 90.0, "sky_brightness": 22.0,           # lenstronomy requirements
        "magnitude_zero_point": 26.0, "num_exposures": 50,
        "seeing": 0.5, "psf_type": "GAUSSIAN",
        "cadence_days": 20.0                                      # band-specific cadence
    },
    "B": {
        "wave_center": 5000.0, "wave_width": 1000.0,
        "exposure_time": 90.0, "sky_brightness": 21.5,
        "magnitude_zero_point": 26.0, "num_exposures": 50,
        "seeing": 0.5, "psf_type": "GAUSSIAN",
        "cadence_days": 20.0
    },
    "C": {
        "wave_center": 6000.0, "wave_width": 1000.0,
        "exposure_time": 90.0, "sky_brightness": 21.0,
        "magnitude_zero_point": 26.0, "num_exposures": 50,
        "seeing": 0.5, "psf_type": "GAUSSIAN",
        "cadence_days": 20.0
    },
    "D": {
        "wave_center": 7000.0, "wave_width": 1000.0,
        "exposure_time": 90.0, "sky_brightness": 20.5,
        "magnitude_zero_point": 26.0, "num_exposures": 50,
        "seeing": 0.5, "psf_type": "GAUSSIAN",
        "cadence_days": 20.0
    },
}

MIDEX_CAMERA = {
    "read_noise": 5.0,     
    "pixel_scale": 0.15,   
    "ccd_gain": 2.0,       
}


# ==========================================
# 2. SPECLITE FILTER GENERATOR
# ==========================================
def get_midex_band_names(prefix="MidEx-"):
    """Returns a list of registered filter names (e.g., ['MidEx-A', ...])"""
    return [f"{prefix}{band}" for band in MIDEX_BANDS.keys()]

def configure_midex_filters(save_path=None):
    """Generates top-hat filters from the master dict and registers them in speclite."""
    group_name = "MidEx"

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for band, props in MIDEX_BANDS.items():
        min_wave = props["wave_center"] - (props["wave_width"] / 2.0)
        max_wave = props["wave_center"] + (props["wave_width"] / 2.0)
        
        wave = np.linspace(min_wave - 100, max_wave + 100, 1000)
        wavelength = wave * u.Angstrom
        
        response = np.zeros_like(wave)
        mask = (wave >= min_wave) & (wave <= max_wave)
        response[mask] = 1.0  

        speclite_filter = speclite.filters.FilterResponse(
            wavelength=wavelength,
            response=response,
            meta=dict(group_name=group_name, band_name=band),
        )
        
        if save_path is not None:
            speclite_filter.save(save_path)


# ==========================================
# 3. LENSTRONOMY OBSERVATION CLASS
# ==========================================
class MidEx(object):
    """Class contains MidEx instrument and observation configurations for Lenstronomy."""

    def __init__(self, band="A", psf_type="GAUSSIAN", coadd_years=10):
        if band.isalpha():
            band = band.upper()
            
        if band not in MIDEX_BANDS:
            raise ValueError(f"band {band} not supported! Choose from {list(MIDEX_BANDS.keys())}.")
            
        lenstronomy_keys = ["exposure_time", "sky_brightness", "magnitude_zero_point", "num_exposures", "seeing", "psf_type"]
        self.obs = {k: MIDEX_BANDS[band][k] for k in lenstronomy_keys}
        
        if psf_type != "GAUSSIAN":
            raise ValueError(f"psf_type {psf_type} not supported!")

        if coadd_years > 10 or coadd_years < 1:
            raise ValueError(f"{coadd_years} coadd_years not supported! Choose an integer between 1 and 10.")
        elif coadd_years != 10:
            self.obs["num_exposures"] = coadd_years * self.obs["num_exposures"] // 10

        self.camera = copy.deepcopy(MIDEX_CAMERA)

    def kwargs_single_band(self):
        return util.merge_dicts(self.camera, self.obs)

# ==========================================
# 4. SKYPY YAML CONFIG GENERATOR
# ==========================================
def generate_skypy_config(output_path="configs/roman-lsst-MidEx-like.yml"):
    """
    Generates a dynamic SkyPy configuration file incorporating MidEx, Roman, and LSST bands
    based on the MIDEX_BANDS dictionary.
    """
    # Define base filters and mags
    roman_filters = ['Roman-F062', 'Roman-F087', 'Roman-F106', 'Roman-F129', 'Roman-F158', 'Roman-F184', 'Roman-F146', 'Roman-F213']
    roman_mags = ['mag_F062', 'mag_F087', 'mag_F106', 'mag_F129', 'mag_F158', 'mag_F184', 'mag_F146', 'mag_F213']

    lsst_filters = ['lsst2016-g', 'lsst2016-r', 'lsst2016-i', 'lsst2016-z', 'lsst2016-y']
    lsst_mags = ['mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_y']

    # Construct dynamic MidEx strings
    midex_band_keys = list(MIDEX_BANDS.keys())
    midex_filters = [f"MidEx-{b}" for b in midex_band_keys]
    midex_mags = [f"mag_{b}" for b in midex_band_keys]

    # Combine them all
    all_filters = roman_filters + midex_filters + lsst_filters
    all_mags = roman_mags + midex_mags + lsst_mags

    # Format them for the YAML file
    filter_str = ", ".join([f"'{f}'" for f in all_filters])
    mag_str = ", ".join(all_mags)

    yaml_content = f"""mag_lim: 30
fsky: 0.1 deg2
z_range: !numpy.arange [0.0, 5.01, 0.01]
M_star_blue: !astropy.modeling.models.Linear1D [-0.9408582, -20.40492365]
phi_star_blue: !astropy.modeling.models.Exponential1D [0.00370253, -9.73858]
alpha_blue: -1.3
M_star_red: !astropy.modeling.models.Linear1D [-0.70798041, -20.37196157]
phi_star_red: !astropy.modeling.models.Exponential1D [0.0035097, -1.41649]
alpha_red: -0.5
cosmology: !astropy.cosmology.default_cosmology.get []

# --- Dynamically Generated Filters ---
filters: [{filter_str}]

tables:
  blue:
    z: !skypy.galaxies.redshift.schechter_lf_redshift
      redshift: $z_range
      M_star: $M_star_blue
      phi_star: $phi_star_blue
      alpha: $alpha_blue
      m_lim: $mag_lim
      sky_area: $fsky
      cosmology: $cosmology
    M: !skypy.galaxies.luminosity.schechter_lf_magnitude
      redshift: $blue.z
      M_star: $M_star_blue
      alpha: $alpha_blue
      m_lim: $mag_lim
    coeff: !skypy.galaxies.spectrum.dirichlet_coefficients
      redshift: $blue.z
      alpha0: [2.079, 3.524, 1.917, 1.992, 2.536]
      alpha1: [2.265, 3.862, 1.921, 1.685, 2.480]
      weight: [3.47e+09, 3.31e+06, 2.13e+09, 1.64e+10, 1.01e+09]
    stellar_mass: !skypy.galaxies.spectrum.kcorrect.stellar_mass
      magnitudes: $blue.M
      coefficients: $blue.coeff
      filter:  bessell-B
      
    # --- Dynamically Generated Magnitude Output Columns ---
    {mag_str}: !skypy.galaxies.spectrum.kcorrect.apparent_magnitudes
      coefficients: $blue.coeff
      filters: $filters
      redshift: $blue.z
      stellar_mass: $blue.stellar_mass
      cosmology: $cosmology
      
    physical_size: !skypy.galaxies.morphology.late_type_lognormal_size
      magnitude: $blue.M
      alpha: 0.21
      beta: 0.53
      gamma: -1.31
      M0: -20.52
      sigma1: 0.48
      sigma2: 0.25
    angular_size: !skypy.galaxies.morphology.angular_size
      physical_size: $blue.physical_size
      redshift: $blue.z
      cosmology: $cosmology
    ellipticity: !skypy.galaxies.morphology.beta_ellipticity
      e_ratio: 0.45
      e_sum: 3.5
      size: !len [$blue.z]
      
  red1: 
    z1, M1: !skypy.galaxies.schechter_lf
              redshift: $z_range
              M_star: $M_star_red
              phi_star: $phi_star_red
              alpha: -0.53
              m_lim: $mag_lim
              sky_area: $fsky
              cosmology: $cosmology
  red2:  
    z2, M2: !skypy.galaxies.schechter_lf
              redshift: $z_range
              M_star: -17.00
              phi_star: $phi_star_red
              alpha: -1.31
              m_lim: $mag_lim
              sky_area: $fsky
              cosmology: $cosmology
  red:
    z: !numpy.concatenate
      - [$red1.z1, $red2.z2]
    M: !numpy.concatenate
      - [$red1.M1, $red2.M2]
    coeff: !skypy.galaxies.spectrum.dirichlet_coefficients
      redshift: $red.z
      alpha0: [2.461, 2.358, 2.568, 2.268, 2.402]
      alpha1: [2.410, 2.340, 2.200, 2.540, 2.464]
      weight: [3.84e+09, 1.57e+06, 3.91e+08, 4.66e+10, 3.03e+07]
    stellar_mass: !skypy.galaxies.spectrum.kcorrect.stellar_mass
      magnitudes: $red.M
      coefficients: $red.coeff
      filter:  bessell-B
      
    # --- Dynamically Generated Magnitude Output Columns ---
    {mag_str}: !skypy.galaxies.spectrum.kcorrect.apparent_magnitudes
      coefficients: $red.coeff
      filters: $filters
      redshift: $red.z
      stellar_mass: $red.stellar_mass
      cosmology: $cosmology
      
    physical_size: !skypy.galaxies.morphology.early_type_lognormal_size
      magnitude: $red.M
      a: 0.60
      b: -4.63
      M0: -20.52
      sigma1: 0.48
      sigma2: 0.25
    angular_size: !skypy.galaxies.morphology.angular_size
      physical_size: $red.physical_size
      redshift: $red.z
      cosmology: $cosmology
    ellipticity: !skypy.galaxies.morphology.beta_ellipticity
      e_ratio: 0.2
      e_sum: 7
      size: !len [$red.z]
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_content)
    
    print(f"SkyPy config successfully generated at: {output_path}")

  
# ==========================================
# 5. MIDEX TIME SAMPLING (CADENCE)
# ==========================================
def get_midex_cadence_dict(duration_days=3650, starting_time_offset=0):
    """
    Generates a mock observation cadence for MidEx bands based on the master dictionary.
    """
    cadence_dict = {}
    
    for band, props in MIDEX_BANDS.items():
        cadence = props.get("cadence_days", 5.0) # Fallback to 5 if not defined
        
        # Generates evenly spaced observations over the duration
        times = np.arange(0, duration_days, cadence) + starting_time_offset
        
        # Add a tiny bit of random jitter (+/- 0.5 days) for realism
        jitter = np.random.uniform(-0.5, 0.5, size=len(times))
        cadence_dict[band] = times + jitter
        
    return cadence_dict