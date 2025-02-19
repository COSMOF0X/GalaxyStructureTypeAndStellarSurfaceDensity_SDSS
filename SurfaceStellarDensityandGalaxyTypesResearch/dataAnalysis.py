import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# Grab the spiral and elliptical data from the CSV files:
spiralGalDataCSV = pd.read_csv('SkyserverData/SkyServerDR18_SpiralStellarMassandRadius.csv', skiprows = 1)
ellipticalGalDataCSV = pd.read_csv('SkyserverData/SkyServerDR18_EllipticalStellarMassandRadius.csv', skiprows = 1)

# Eliminate any whitespace in the column names in the CSV files:
spiralGalDataCSV.columns = spiralGalDataCSV.columns.str.strip()
ellipticalGalDataCSV.columns = ellipticalGalDataCSV.columns.str.strip()

# Unpack spiral galaxy data:
spiralSloanID = spiralGalDataCSV['ID'].to_numpy()
spiralRA = spiralGalDataCSV['RA_deg'].to_numpy()
spiralDEC = spiralGalDataCSV['DEC_deg'].to_numpy()
spiralZ = spiralGalDataCSV['Redshift'].to_numpy()
spiralMedStelMassRaw = spiralGalDataCSV['MedianStellarMass_dex_solarmasses'].to_numpy()
spiralMedStelMass = np.power(10, spiralMedStelMassRaw)
spiralExpRadRBand = spiralGalDataCSV['ExponentialGalRad_bandR_arcsec'].to_numpy()

# Unpack spiral galaxy data:
ellipticalSloanID = ellipticalGalDataCSV['ID'].to_numpy()
ellipticalRA = ellipticalGalDataCSV['RA_deg'].to_numpy()
ellipticalDEC = ellipticalGalDataCSV['DEC_deg'].to_numpy()
ellipticalZ = ellipticalGalDataCSV['Redshift'].to_numpy()
ellipticalMedStelMassRaw = ellipticalGalDataCSV['MedianStellarMass_dex_solarmasses'].to_numpy()
ellipticalMedStelMass = np.power(10, ellipticalMedStelMassRaw)
ellipticalDeVanRadRBand = ellipticalGalDataCSV['DeVancoulerGalRad_bandR_arcsec'].to_numpy()

# Create galaxy morphology type array:
galMorphTypeNames = ["Spiral\n(Sample Size: 10,000 Galaxies)", "Elliptical\n(Sample Size: 10,000 Galaxies)"]
galMorphTypeNums = [1, 2]

# Functions for data analysis:
def stelSurfDen(stelMass, rad):
    # SD* = M* / π(R^2)
    return (stelMass / (math.pi * (rad)**2))

def stdError(data):
    # SE = σ / √(N)
    return ((np.std(data, ddof = 1)) / ((data.size)**0.5))

def analyzeStelSurfDenandGalType(galTypeNums, galTypeNames, stelMassSpiral, radSpiral, stelMassElliptical, radElliptical):
    # Spiral:
    stelSurfDenDataSpiral = stelSurfDen(stelMassSpiral, radSpiral)
    stelSurfDenAvgSpiral = np.mean(stelSurfDenDataSpiral)
    stelSurfDenErrorSpiral = stdError(stelSurfDenDataSpiral)

    # Elliptical:
    stelSurfDenDataElliptical = stelSurfDen(stelMassElliptical, radElliptical)
    stelSurfDenAvgElliptical = np.mean(stelSurfDenDataElliptical)
    stelSurfDenErrorElliptical = stdError(stelSurfDenDataElliptical)

    # group Spiral and Elliptical:
    avgStelSurfDen = [stelSurfDenAvgSpiral, stelSurfDenAvgElliptical]
    stelSurfDenError = [stelSurfDenErrorSpiral, stelSurfDenErrorElliptical]

    # Display Data
    plt.figure()
    plt.errorbar(galTypeNums[0], avgStelSurfDen[0], yerr = stelSurfDenError[0], fmt = 'o', markersize = 8)
    plt.errorbar(galTypeNums[1], avgStelSurfDen[1], yerr = stelSurfDenError[1], fmt = 'o', markersize = 8)

    plt.xticks(galTypeNums, galTypeNames)
    plt.yscale('log')
    plt.xlabel("Galaxy Morphological Type")
    plt.ylabel("Average Stellar Surface Density (M☉/(arcsecond^2))")
    plt.title("Galaxy Average Stellar Surface Density and Morphological Type\n(Derived from SDSS DR18)")
    plt.show()

    plt.savefig("avgStelSurfDenandGalMorphType.png")
    return

analyzeStelSurfDenandGalType(galMorphTypeNums, galMorphTypeNames, spiralMedStelMass, spiralExpRadRBand, ellipticalMedStelMass, ellipticalDeVanRadRBand)