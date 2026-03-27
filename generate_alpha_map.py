
import copy
from HETS import HETS

from Utilities.Constants import PA_PER_PSI


# --- Inputs ---
filename = "alpha_map.parquet"
test_stand = HETS


# --- Generate alpha map ---
ts = copy.deepcopy(test_stand)

pcmr_map = ts.generate_PcMR_map(
    MR_target=2.0,
    Pc_min=250 * PA_PER_PSI,            # Lower bound Pc on map
    Pc_max=310 * PA_PER_PSI,            # Upper bound Pc on map
    Pc_step=0.1 * PA_PER_PSI,             # dPc on map
    fuel_CdA_range=(1.0e-6, 1.5e-4),    # steady state solver limits on Fuel side
    ox_CdA_range=(1.0e-6, 1.5e-4),      # steady state solver limits on Ox side
    return_dataframe = True,            # If you want the map as a pandas dataframe
    save_parquet = True,                # Make this True if you want a parquet file
    parquet_filename = filename,        # parquet file names
    verbose=False,                      # Set this to true if you want to see if the generation failed
)

