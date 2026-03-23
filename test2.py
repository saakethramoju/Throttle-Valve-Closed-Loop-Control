from Utilities import set_winplot_dark, solve_system_CdAs, get_density
from Physics import PA_PER_PSI, M2_PER_IN2, N_PER_LBF

print(get_density('n-Dodecane', 400 * PA_PER_PSI, 300.000001))
print(get_density('RP-1', 400 * PA_PER_PSI, 300.000001))