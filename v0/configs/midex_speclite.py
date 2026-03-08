import numpy as np
import os
from astropy import units as u
import speclite.filters

# Define the properties of your top-hat filters. 
# Format: "Band_Name": (Center_Wavelength_Angstroms, Width_Angstroms)
_filter_specs = {
    "A": (4000.0, 1000.0),
    "B": (5000.0, 1000.0),
    "C": (6000.0, 1000.0),
    "D": (7000.0, 1000.0),
}

def filter_names():
    """
    :return: list of full registry names for the MidEx filters
    """
    return [f"MidEx-{name}" for name in _filter_specs.keys()]

def configure_midex_filters(save_path=None):
    """
    Generates top-hat filters and registers them in speclite.
    
    :param save_path: Optional directory to save the .ecsv files. 
                      If None, they are just kept in memory for the session.
    :return: MidEx filters stored and accessible in speclite
    """
    group_name = "MidEx"

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for filter_name, (center, width) in _filter_specs.items():
        # 1. Define boundaries
        min_wave = center - (width / 2.0)
        max_wave = center + (width / 2.0)
        
        # 2. Create wavelength array with padding so it tapers to 0 at the ends
        # Using 1000 points for smooth edges when plotting
        wave = np.linspace(min_wave - 100, max_wave + 100, 1000)
        wavelength = wave * u.Angstrom
        
        # 3. Create top-hat response
        response = np.zeros_like(wave)
        mask = (wave >= min_wave) & (wave <= max_wave)
        response[mask] = 1.0  # 100% transmission inside the top-hat

        # 4. Instantiate the filter. 
        # *CRITICAL*: This step automatically registers it in speclite's memory for this session!
        speclite_filter = speclite.filters.FilterResponse(
            wavelength=wavelength,
            response=response,
            meta=dict(group_name=group_name, band_name=filter_name),
        )
        
        # 5. Optionally save to disk (like in your Roman code)
        if save_path is not None:
            speclite_filter.save(save_path)