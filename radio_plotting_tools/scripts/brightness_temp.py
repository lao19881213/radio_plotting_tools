import numpy as np

import pandas as pd
import uncertainties.unumpy as unp
import scipy.constants as const
import astropy.units as u

df_components = pd.read_csv('components.csv')

lamb = const.c / 15e9
kB = unp.uarray(const.physical_constants['Boltzmann constant'][0], const.physical_constants['Boltzmann constant'][2])
S = (df_components.loc[df_components['c_i'] == 0, 'flux'])
z = 0.078
a_maj = u.mas.to(u.rad, df_components.loc[df_components['c_i'] == 0, 'major_axes'])
a_min = u.mas.to(u.rad, df_components.loc[df_components['c_i'] == 0, 'minor_axes'])


bright_temp = (2 * np.log(2) / (np.pi * kB)) * ( (S*1e-26 * lamb**2 * (1 + z)) / ( a_maj * a_min ))

df_bright_temp = pd.DataFrame({'bright_temp': bright_temp})

print(df_bright_temp)

pd.concat([df_components, df_bright_temp], axis=1)

print(df_components)


'''
cross check: core 2007-11-10 table 3.7

lamb = const.c / 8.4e9
print((2 * np.log(2) / (np.pi * kB)) * ( (1.29e-26 * lamb**2 * (1 + 0.056)) / ((0.126*u.mas).to(u.rad) * (0.066*u.mas).to(u.rad) )) )

'''