# import numpy as np
# import pandas as pd
# import matplotlib as plt

# SpiralGalData = pd.read_csv('SkyserverData/SkyServerDR18_SpiralStellarMassandRadius.csv')
# EllipticalGalData = pd.read_csv('SkyserverData/SkyServerDR18_EllipticalStellarMassandRadius.csv')

# # SpiralGalData = np.genfromtxt('SkyserverData/SkyServerDR18_SpiralStellarMassandRadius.csv', delimiter=',', names=True, dtype=None, encoding=None)
# # EllipticalGalData = np.genfromtxt('SkyserverData/SkyServerDR18_EllipticalStellarMassandRadius.csv', delimiter=',', names=True, dtype=None, encoding=None)

# spiral_array = SpiralGalData.to_numpy()
# elliptical_array = EllipticalGalData.to_numpy()

# # Access the first RA_deg value from the spiral galaxy data
# first_ra_deg = spiral_array[0, SpiralGalData.columns.get_loc('RA_deg')]
# print(first_ra_deg)

import pandas as pd
import numpy as np

# Load the data from the CSV file using pandas, skipping the first line
spiral_df = pd.read_csv('SkyserverData/SkyServerDR18_SpiralStellarMassandRadius.csv', skiprows=1)
elliptical_df = pd.read_csv('SkyserverData/SkyServerDR18_EllipticalStellarMassandRadius.csv', skiprows=1)

# Print the column names to verify they match the expected names
print(spiral_df.columns)

# Strip any leading/trailing spaces from the column names
spiral_df.columns = spiral_df.columns.str.strip()

# Convert pandas DataFrame to numpy array
spiral_array = spiral_df.to_numpy()
elliptical_array = elliptical_df.to_numpy()

# Access the first RA_deg value from the spiral galaxy data
first_ra_deg = spiral_array[0, spiral_df.columns.get_loc('RA_deg')]
print(first_ra_deg)
