# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 01:47:13 2021

@author: Hiroki Kozuki
"""
# Year 1 Computing Project 1 - Calculating the Hubble Constant
# Note: Whenever I mention standard deviation, I am refering to sample standard deviation.

# Section 1: Loading data files.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# load data file for H-alpha spectral lines excluding row with frequency values.
spectral_line = np.loadtxt(r'C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 1\Y1 Term 1\Y1 Computing Lab\Computing Project 1 - Hubble Constant\CompProjectData\Halpha_spectral_data.csv', skiprows=5, delimiter=',') 
# load data file for galaxy distances.
distance = np.loadtxt(r'C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 1\Y1 Term 1\Y1 Computing Lab\Computing Project 1 - Hubble Constant\CompProjectData\Distance Data.csv', skiprows=1, delimiter=',') 

#%%

# Section 2: 
    # a) Sorting data arrays.
    # b) Discarding corrupt data (rows with response == 0):

        
# a) Sorting data arrays:        

# Sort spectral_line and distances data arrays in terms of osbervation number. 
# Reference for np.argsort(): https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column

# Sort spectral line data in ascending order of observation number.
sorted_spectral_line = spectral_line[spectral_line[:, 0].argsort()] 

# Sort distance data in ascending order of observation number.
sorted_distance = distance[distance[:, 0].argsort()] 


# b) Discarding corrupt data (rows with response == 0):

# Concatenate sorted_spectral_line and sorted_distance data such that response column comes last. 
# Reference for np.concatenate(): https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
concatenated_array = np.concatenate((sorted_spectral_line, sorted_distance), axis=1)

# Create new array by appending rows whose final column element (response) == 1 into a new, initially empty array. 
# Reference for np.append(): https://numpy.org/doc/stable/reference/generated/numpy.append.html
clean_conc_array = [] 

# Then, create for-loop for each row in the concatenated array:
for i in range(len(concatenated_array)): 
    if float(concatenated_array[i,-1]) == 1: # for every row with last column element == 1, append the row to clean_conc_array.
        clean_conc_array.append(concatenated_array[i])

# Stack collection of 1D arrays as rows to create 2D array using np.vstack().
# Reference for np.vstack(): https://numpy.org/devdocs/reference/generated/numpy.row_stack.html
clean_conc_array = np.vstack(clean_conc_array)  

print(len(clean_conc_array)) # check that the number of rows in new array is 25.
    
# Obtain clean spectral line data array by slicing every columns except 3 last columns. It contains 25 rows whose 1st column is the observation number and the remaining columns are spectral line values.
clean_spectral_line = clean_conc_array[:,:-3] 
# Obtain clean distance data array by slicing last 3 columns. It contains 25 rows whose 1st column is the observation number, 2nd the distance, and 3rd the response number (1). 
clean_distance = clean_conc_array[:,-3:] 


# Obtain 1D array of frequencies to plot against spectral line data:
frequency_plus_spectral_line = np.loadtxt(r'C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 1\Y1 Term 1\Y1 Computing Lab\Computing Project 1 - Hubble Constant\CompProjectData\Halpha_spectral_data.csv', skiprows=4, delimiter=',') 
# Define frequency array as the first row of the delimited data set starting from 2nd column.
frequency = frequency_plus_spectral_line[0,1:] 


#%%

# Section 3: 
    # a) Defining linear gaussian function:
    # b) Create for-loop to generate initial guesses for each plot (except standard deviation, which was determined manually):
        # 1) Create Initial guess for m, c, mu, a, sig.
        # 2) Create Curve_fit for 25 plots of Frequency (Hz) (independent variable) vs. Spectral Line Intensity (arbitrary units) (dependent variable).
        # 3) Calculate Red shifted wavelength value and its uncertainty (standard deviation) for each plot/observation number.
        

# a) Defining linear gaussian function:

# Variables and parameters:
    # frequency = frequency of each observed spectral line. (Independent variable)
    # a = amplitude of gaussian region of fit.
    # mu = mean frequency value at which the maximum (amplitude) of the gaussian region of fit occurs.
    # sig = standard deviation in the gaussian region of fit.
    # m = slope of linear region of fit.
    # c = y-intercept of linear region of fit.
# Reference for linear gaussian function: Imperial College London Physics 1st Year Computing Lab Core Worksheet 2 Solutions.
def lin_gauss(frequency,a,mu,sig,m,c):
    lin = m*frequency+c
    gauss = a*np.exp(-(frequency-mu)**2/(2*sig**2))
    return lin + gauss


# Empty arrays for later use:
λo = []
λo_std = []


# b) Create for-loop to generate initial guesses for each plot (except standard deviation):


# Set range to iterates for the number of rows in clean_spectral_line data set (25):

for i in range(len(clean_spectral_line)):
    
    # 1) Initial guesses:
    
    # After trying the method suggested in the computing project lecture for initial guess for m (divide the difference of last and first spectral line values with the range in frequency values),
    # I decided to improve its quality by fitting linear fit through the entire plot and using its slope and y-intercept as initial guesses for m and c.
    # Create linear fit for each data set.
    fit_lin = np.polyfit(frequency,clean_spectral_line[i,1:],1)

    # Initial guess for m is the slope of the linear fit.
    m_guess = fit_lin[0]
    
    # Initial guess for c is the y-intercept of the linear fit.
    c_guess = fit_lin[1]
    
    # Find the residuals (differences) between spectral line data points and the values predicted by the crude linear fit (determined using m_guess and c_guess) for all frequency values.
    residual = clean_spectral_line[i,1:] - m_guess*frequency - c_guess
  
    # Find the index of frequency value at which the difference between the spectral line value and linear fit is a maximum.
    # Source for np.amax(): https://www.google.com/search?q=numpy+maximum+of+array&oq=numpy+maximum+&aqs=chrome.1.69i57j0i512l9.3874j0j7&sourceid=chrome&ie=UTF-8
    mean_freq_index = np.where(residual == np.amax(residual))
    # There are several peaks whose residuals have the same value. Hence choose the first entry in the array.
    single_mean_freq_index = mean_freq_index[0]
    single_mean_freq_index = single_mean_freq_index[0] # This line is for debugging. Somehow mean_freq_index[0] came out as 2D array.

    # initial guess of the mean of the gaussian will be the frequency value corresponding to the single_mean_freq_index above, i.e. frequency corresponding to the gaussian peak. 
    mu_guess = frequency[single_mean_freq_index]

    # To find initial guess for amplitude of gaussian, shift the spectral line plot downwards by subtracting the corresponding values of linear fit from the spectral line values.
    # I can then take the maximum y-value of the new downwardly shifted plot as my initial guess of amplitude. 
    # Reference for np.subtract(): https://numpy.org/doc/stable/reference/generated/numpy.subtract.html
    down_shifted_gaussian = np.subtract(clean_spectral_line[i,1:], m_guess*frequency + c_guess)
    a_guess = np.amax(down_shifted_gaussian)
        
    # Find initial guess of standard deviation by manually adjusting its order of magniutde until clear curve fits are obtained for all 25 plots (as judged by eye).
    # Manual trial and error yields 10**12 as the optimum achievable initial guess for standard deviation.
    sig_guess = 10**12 # Hz
    # This manual guess method was inspired by my collegue Yaar Safra.
    
    # Compile initial guesses in a 1D array to be substituted into curve_fit function altogether.
    initial_guess = [a_guess, mu_guess, sig_guess, m_guess, c_guess]
    
    
    # 2) Overplot curve_fit on 25 plots:
    
    # Apply curve fit (lin_gauss):
    # References for scipy.optimize.curve_fit(): 
    # 1) https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # 2) https://stackoverflow.com/questions/34978695/what-does-maxfev-do-in-ipython-notebook
    parameters, parameters_cov = curve_fit(lin_gauss, frequency, clean_spectral_line[i,1:], initial_guess, maxfev=1000000)
    # Here, the updated parameters (a,mu,sig,m,c) are stored in parameters array, and thier covariance matrix in parameters_cov (from which we can determine their standard deviations).
    # Maxfev is the maximum number of iteration the command executes to obtain better parameter values. I set this to an arbitrarily high number.

    # Overplot linear gaussian curve_fit on 25 spectral shift plots. Use subplot to display 25 graphs simultaneously in rows of five.
    plt.subplot(5,5,i+1)
    plt.plot(frequency,clean_spectral_line[i,1:])                    # Plot frequency (Hz) (x-axis) against Spectral line/shift intensity (arbitrary units) (y-axis)
    plt.plot(frequency,lin_gauss(frequency,*parameters))             # Overplot curve_fit.
    plt.xlabel('Frequency (Hz)', fontsize = 4)                       # Label x-axis 
    plt.ylabel('Spectral Line Intensity (arb units)', fontsize = 3)  # Label y-axis
    plt.title(clean_distance[i,0], fontsize = 6)                     # Title each plot as its corresponding observation number.
    
    
    # 3) Determine and store red shifted (observed) wavelengths and their uncertainties:

    # Red shifted (observed) frequency values are defined by the second entry in the parameters array.
    # Hence, the observed wavelength values (λo) are defined by the speed of light divided by second entry in parameters array.
    # Append each λo value into an empty array made earlier:
    # Reference for sp.constant.c: https://www.w3schools.com/python/scipy/scipy_constants.php
    λo.append(sp.constants.c/parameters[1])
    
    # The uncertainty (standard deviation) in the updated value of mu (mean frequency at which gaussian peak occurs) will be used as the uncertainty in red shifted frequency.
    # This is the suqare root of the 2nd diagonal entry in the parameters covariance matrix.
    # Standard deviation in red shift frequency will be propagated in quadrature to give standard deviation in λo.
    # Append standard deviation in λo to an empty array made earlier.
    λo_std.append(np.sqrt((sp.constants.c*np.sqrt(parameters_cov[1,1])/(parameters[1]**2))**2))  
    
    # End of for-loop
              
# Show plot (plot was saved as png directly from Plots console)
plt.show()


#%%

# Section 4: Calculate the velocities of receding galaxy and their uncertainties (standard deviations).

λe = 656.28*10**-9 # Wavelength of emitted Hydrogen-alpha spectral line (m). Reference: Imperial College London Year 1 Computing Project 1 Outline.
# convert lists to numpy arrays such that imputs are valid in proceeding lines of code:
λo = np.asarray(λo)
λo_std = np.asarray(λo_std)
# Now, create separate 1D arrays of redshifted velocities and its standard deviations.
# Rearranging formula in coding project outline yeilds the following expression for red shifted velocity.
v_red_shift = (sp.constants.c*(((λo)**2-λe**2)/(((λo)**2)+λe**2)))/1000 # Divide by 1000 to get values in units of km/s
# Obtain standard deviation in red shift velocity by propagating standard devaition in observed λo using quadrature method:
v_red_shift_std = (np.sqrt((sp.constants.c*((λo_std)**2)*(4*(λe**2)*λo)/((λo)**2+λe**2)**2)**2))/1000


#%%

# Section 5: Plot graph of Galaxy distance from Earth (Mpc) (x-axis) vs. Red shifted velocity (km/s) (y-axis), and overplot linear best fit:

# Generate linear fit coefficients and their covariance matrix.
# Increase the accuracy of best fit line by including weightings for each red shifted velocity value as 1 over its standard deviation.
# Convert any lists to arrays to make sure they are valid imputs. 
fit_lin_Hubble, fit_lin_Hubble_cov = np.polyfit(clean_distance[:,1], np.asarray(v_red_shift), 1, w = 1/np.asarray(v_red_shift_std), cov = True) 

# Generate polyfit to be overplotted.
polyfit_Hubble = np.poly1d(fit_lin_Hubble)

# Overplot linear best fit line:
plt.plot(clean_distance[:,1], polyfit_Hubble(clean_distance[:,1]))

# Plot galaxy distance from Earth (Mpc) vs. Red shifted velocity (km/s) with vertical error bars:
plt.errorbar(clean_distance[:,1], v_red_shift, yerr = v_red_shift_std, fmt = 'o', mew = 2, ms = 3, capsize = 4)
plt.title('Galaxy Distance from Earth (Mpc) vs. Red Shift Velocity (km/s)') # Title graph
plt.xlabel('Galaxy Distance from Earth (Mpc)')                              # Label x-axis
plt.ylabel('Red Shift Velocity (km/s)')                                     # Label y-axis
plt.grid()                                                                  # show grid on graph
plt.show()                                                                  # Show graph: (Graph was saved as png directly from Plots console)


#%%

# Section 6: Calculate Hubble Constant and its uncertainty (standard error on the mean) from linear fit obtained in section 5.

# Hubble constant is the first entry in the fit_lin_Hubble array of fit coefficients, which is the slope of the linear best fit.
Hubble_constant = fit_lin_Hubble[0]
# Calculate standard deviation in Hubble constant by taking the square root of the first diagonal entry in the fit_lin_Hubble_cov covariance matrix:
Hubble_constant_std = np.sqrt(fit_lin_Hubble_cov[0,0])
# Calculate standard error of the mean of Hubbble constant by dividing its standard deviation by the square root of the number of data points used to create linear best fit (25):
Hubble_constant_sem = Hubble_constant_std/np.sqrt(len(v_red_shift))

# Print results:
# std and sem shall be quoted to first significant figure.
print('Calculated Hubble constant = %d +/- %d Km/sMpc' %(Hubble_constant, Hubble_constant_sem))
print('Standard deviation in calculated Hubble constant = +/- %d Km/sMpc' %(Hubble_constant_std))


# END OF CODE
    
    
    
    
    
    
    