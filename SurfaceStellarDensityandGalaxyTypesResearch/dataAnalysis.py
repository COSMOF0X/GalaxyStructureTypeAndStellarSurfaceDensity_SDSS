import math
from scipy import stats
from astropy.cosmology import Planck18 as p18
import pandas as pd
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
galMorphTypeNames1 = np.array(["Spiral\n(Sample Size: 10,000 Galaxies)", "Elliptical\n(Sample Size: 10,000 Galaxies)"])
galMorphTypeNames2 = np.array(["Spiral (Sample Size: 10,000 Galaxies)", "Elliptical (Sample Size: 10,000 Galaxies)"])


# Functions for calculations of data:
def stelSurfDen(stelMass, physRad):
    # SD* = M* / π(R^2)
    return (stelMass / (math.pi * (physRad)**2))

def stdError(data):
    # SE = σ / √(N)
    return ((np.std(data, ddof = 1)) / ((data.size)**0.5))

def galRadKPC(redshift, arcsecRad):
    # R (in units kpc) = Da * θ * 1000
    angDiamDist = p18.angular_diameter_distance(redshift).value

    radianRad = arcsecRad / 206264.8062471

    return ((angDiamDist * radianRad) * 1000)

def tTest(groups):
    # Use T-Test to find out how statistically significant the difference is between two pieces of data.
    tStat, pVal = stats.ttest_ind(groups[0], groups[1])

    return np.array([tStat, pVal])

def correlTest(x, y):
    # Use the Correlation Test to find out how statistically significant and strong the relationship is between two variables.
    correlCoeff, pVal = stats.spearmanr(x, y)

    return np.array([correlCoeff, pVal])

# Functions for data analysis:
def analyzeAvgStelSurfDenandGalType(galTypeNames, stelMassSpiral, radSpiral, zSpiral, stelMassElliptical, radElliptical, zElliptical):
    # Spiral:
    radPhysSpiral = galRadKPC(zSpiral, radSpiral)
    stelSurfDenDataSpiral = stelSurfDen(stelMassSpiral, radPhysSpiral)
    stelSurfDenAvgSpiral = np.mean(stelSurfDenDataSpiral)
    stelSurfDenErrorSpiral = stdError(stelSurfDenDataSpiral)

    # Elliptical:
    radPhysElliptical = galRadKPC(zElliptical, radElliptical)
    stelSurfDenDataElliptical = stelSurfDen(stelMassElliptical, radPhysElliptical)
    stelSurfDenAvgElliptical = np.mean(stelSurfDenDataElliptical)
    stelSurfDenErrorElliptical = stdError(stelSurfDenDataElliptical)

    # Group spiral and elliptical:
    stelSurfDenDataGrouped = np.array([stelSurfDenDataSpiral, stelSurfDenDataElliptical])
    avgStelSurfDens = np.array([stelSurfDenAvgSpiral, stelSurfDenAvgElliptical])
    stelSurfDenErrors = np.array([stelSurfDenErrorSpiral, stelSurfDenErrorElliptical])

    # T-Test:
    tResult = tTest(stelSurfDenDataGrouped)


    # Set up display parameters:
    barGraphColors = ['cyan', 'orange']
    errorParams = {'ecolor': 'red', 'elinewidth': 2, 'capsize': 15, 'capthick': 2}

    # Display Data:
    plt.clf()
    plt.figure()
    plt.bar(galTypeNames, avgStelSurfDens, yerr = stelSurfDenErrors, color = barGraphColors, edgecolor = 'black', linewidth = 2, error_kw = errorParams)

    plt.yscale('log')
    plt.xlabel("Galaxy Morphological Type", fontweight = 'bold')
    plt.ylabel("Average Stellar Surface Density (M☉/(kpc^2))", fontweight = 'bold')
    plt.title("Galaxy Average Stellar Surface Density and Morphological Type\n(Derived from SDSS DR18)", fontweight = 'bold')

    plt.tight_layout()
    plt.savefig("Graphs/avgStelSurfDenandGalMorphType.png", bbox_inches= 'tight')

    print(f"Average Stellar Surface Density (Spiral): {avgStelSurfDens[0]:.10}")
    print(f"Average Stellar Surface Density (Elliptical): {avgStelSurfDens[1]:.10}")
    print(f"T-Test T-Statistic: {tResult[0]:.10e}")
    print(f"T-Test P-Value: {tResult[1]:.10e}")

    return

def analyzeAvgMedStelMassandGalType(galTypeNames, stelMassSpiral, stelMassElliptical):
    # Spiral:
    stelMassAvgSpiral = np.mean(stelMassSpiral)
    stelMassErrorSpiral = stdError(stelMassSpiral)

    # Elliptical:
    stelMassAvgElliptical = np.mean(stelMassElliptical)
    stelMassErrorElliptical = stdError(stelMassElliptical)

    # Group spiral and elliptical:
    stelMassDataGrouped = np.array([stelMassSpiral, stelMassElliptical])
    avgStelMasses = np.array([stelMassAvgSpiral, stelMassAvgElliptical])
    stelMassErrors = np.array([stelMassErrorSpiral, stelMassErrorElliptical])

    # T-Test:
    tResult = tTest(stelMassDataGrouped)

    # Set up display parameters:
    barGraphColors = ['cyan', 'orange']
    errorParams = {'ecolor': 'red', 'elinewidth': 2, 'capsize': 15, 'capthick': 2}

    # Display Data:
    plt.clf()
    plt.figure()
    plt.bar(galTypeNames, avgStelMasses, yerr = stelMassErrors, color = barGraphColors, edgecolor = 'black', linewidth = 2, error_kw = errorParams)

    plt.yscale('log')
    plt.xlabel("Galaxy Morphological Type", fontweight = 'bold')
    plt.ylabel("Average Stellar Median Mass (M☉)", fontweight = 'bold')
    plt.title("Galaxy Average Stellar Median Mass and Morphological Type\n(Derived from SDSS DR18)", fontweight = 'bold')

    plt.tight_layout()
    plt.savefig("Graphs/avgStelMassandGalMorphType.png", bbox_inches= 'tight')

    print(f"Average Stellar Median Mass (Spiral): {avgStelMasses[0]:.10}")
    print(f"Average Stellar Median Mass (Elliptical): {avgStelMasses[1]:.10}")
    print(f"T-Test T-Statistic: {tResult[0]:.10e}")
    print(f"T-Test P-Value: {tResult[1]:.10e}")

    return

def analyzeAvgGalRadandGalType(galTypeNames, radSpiral, zSpiral, radElliptical, zElliptical):
    # Spiral:
    radPhysSpiral = galRadKPC(zSpiral, radSpiral)
    galRadAvgSpiral = np.mean(radPhysSpiral)
    galRadErrorSpiral = stdError(radPhysSpiral)

    # Elliptical:
    radPhysElliptical = galRadKPC(zElliptical, radElliptical)
    galRadAvgElliptical = np.mean(radPhysElliptical)
    galRadErrorElliptical = stdError(radPhysElliptical)

    # Group spiral and elliptical:
    galRadDataGrouped = np.array([radPhysSpiral, radPhysElliptical])
    avgGalRads = np.array([galRadAvgSpiral, galRadAvgElliptical])
    galRadErrors = np.array([galRadErrorSpiral, galRadErrorElliptical])

    # T-Test:
    tResult = tTest(galRadDataGrouped)

    # Set up display parameters:
    barGraphColors = ['cyan', 'orange']
    errorParams = {'ecolor': 'red', 'elinewidth': 2, 'capsize': 15, 'capthick': 2}

    # Display Data:
    plt.clf()
    plt.figure()
    plt.bar(galTypeNames, avgGalRads, yerr = galRadErrors, color = barGraphColors, edgecolor = 'black', linewidth = 2, error_kw = errorParams)

    plt.yscale('log')
    plt.xlabel("Galaxy Morphological Type", fontweight = 'bold')
    plt.ylabel("Average Galaxy Radius (kiloparsec)", fontweight = 'bold')
    plt.title("Galaxy Average Radius and Morphological Type\n(Derived from SDSS DR18)", fontweight = 'bold')

    plt.tight_layout()
    plt.savefig("Graphs/avgGalRadandGalMorphType.png", bbox_inches= 'tight')

    print(f"Average Stellar Median Mass (Spiral): {avgGalRads[0]:.10}")
    print(f"Average Stellar Median Mass (Elliptical): {avgGalRads[1]:.10}")
    print(f"T-Test T-Statistic: {tResult[0]:.10e}")
    print(f"T-Test P-Value: {tResult[1]:.10e}")

    return

def analyzeStelSurfDenandRedshift(galTypeNames, stelMassSpiral, radSpiral, zSpiral, stelMassElliptical, radElliptical, zElliptical):
    # Spiral:
    radPhysSpiral = galRadKPC(zSpiral, radSpiral)
    stelSurfDenDataSpiral = stelSurfDen(stelMassSpiral, radPhysSpiral)

    # Elliptical:
    radPhysElliptical = galRadKPC(zElliptical, radElliptical)
    stelSurfDenDataElliptical = stelSurfDen(stelMassElliptical, radPhysElliptical)

    # Pearson Correlational Test:
    correlResultSpiral = correlTest(zSpiral, stelSurfDenDataSpiral)
    correlResultElliptical = correlTest(zElliptical, stelSurfDenDataElliptical)

    # Display Data:
    plt.clf()
    figStelSurfDenandZ, axStelSurfDenandZ = plt.subplots(1, 2, figsize = (15, 10), sharex = True, sharey = True)
    axStelSurfDenandZ[0].scatter(zSpiral, stelSurfDenDataSpiral, color = 'cyan', s = 5)
    axStelSurfDenandZ[1].scatter(zElliptical, stelSurfDenDataElliptical, color = 'orange', s = 5)

    axStelSurfDenandZ[0].set_yscale('log')
    axStelSurfDenandZ[0].set_xlabel("Galaxy Redshift (z)", fontweight = 'bold')
    axStelSurfDenandZ[0].set_ylabel("Average Stellar Surface Density (M☉/(arcsecond^2))", fontweight = 'bold')
    axStelSurfDenandZ[0].set_title(f"Morphological Type: {galTypeNames[0]}", fontweight = 'bold')

    axStelSurfDenandZ[1].set_yscale('log')
    axStelSurfDenandZ[1].set_xlabel("Galaxy Redshift (z)", fontweight = 'bold')
    axStelSurfDenandZ[1].set_ylabel("Average Stellar Surface Density (M☉/(arcsecond^2))", fontweight = 'bold')
    axStelSurfDenandZ[1].set_title(f"Morphological Type: {galTypeNames[1]}", fontweight = 'bold')
    
    figStelSurfDenandZ.suptitle("Galaxy Average Stellar Surface Density and Redshift\n(Derived from SDSS DR18)", fontweight = 'bold')

    plt.savefig("Graphs/avgStelSurfDenandGalRedshift.png", bbox_inches= 'tight')

    print(f"Correlation Coefficient (Spiral): {correlResultSpiral[0]}")
    print(f"Correlation P-value (Spiral): {correlResultSpiral[1]:.10e}")
    print(f"Correlation Coefficient (Elliptical): {correlResultElliptical[0]}")
    print(f"Correlation P-value (Elliptical): {correlResultElliptical[1]:.10e}")

    return


# Call function:
print("Average Stellar Surface Density and Galaxy Type:")
analyzeAvgStelSurfDenandGalType(galTypeNames = galMorphTypeNames1, stelMassSpiral = spiralMedStelMass, radSpiral = spiralExpRadRBand, zSpiral = spiralZ, stelMassElliptical = ellipticalMedStelMass, radElliptical = ellipticalDeVanRadRBand, zElliptical = ellipticalZ)

print("\nAverage Stellar Median Mass and Galaxy Type:")
analyzeAvgMedStelMassandGalType(galTypeNames = galMorphTypeNames1, stelMassSpiral = spiralMedStelMass, stelMassElliptical = ellipticalMedStelMass)

print("\nAverage Galaxy Radius and Galaxy Type:")
analyzeAvgGalRadandGalType(galTypeNames = galMorphTypeNames1, radSpiral = spiralExpRadRBand, zSpiral = spiralZ, radElliptical = ellipticalDeVanRadRBand, zElliptical = ellipticalZ)

print("\nStellar Surface Density and Redshift:")
analyzeStelSurfDenandRedshift(galTypeNames = galMorphTypeNames2, stelMassSpiral = spiralMedStelMass, radSpiral = spiralExpRadRBand, zSpiral = spiralZ, stelMassElliptical = ellipticalMedStelMass, radElliptical = ellipticalDeVanRadRBand, zElliptical = ellipticalZ)
