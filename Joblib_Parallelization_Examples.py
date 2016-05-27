
# coding: utf-8

# # Embarisingly Parallel for loops using Joblib
# Example from documentation

# In[1]:

from math import sqrt
[sqrt(i ** 2) for i in range(10)]


# In[2]:

from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))


# The progress meter: the higher the value of verbose, the more messages:
# If verbose > 50 then message a for every task is returned.

# In[3]:

from time import sleep
from joblib import Parallel, delayed
# n_jobs=1 turns off the parallel code for debuging.
r = Parallel(n_jobs=1, verbose=1)(delayed(sleep)(.1) for _ in range(100)) 


# In[4]:

r = Parallel(n_jobs=2, verbose=5)(delayed(sleep)(.1) for _ in range(100)) 


# In[5]:

r = Parallel(n_jobs=-1, verbose=10)(delayed(sleep)(.1) for _ in range(100)) 


# Running the same code on the simple sleep function shows the effect of increasing the number of seperate jobs/processes.

# ## Reusing a pool of workers

# In[6]:

def test_reuse():
    """Test Reusing a pool of workers """
    with Parallel(n_jobs=2) as parallel:
        accumulator = 0.
        n_iter = 0
        while accumulator < 1000:
            results = parallel(delayed(sqrt)(accumulator + i ** 2) for i in range(5))
            accumulator += sum(results)  # synchronization barrier
            n_iter += 1


# In[7]:

def test_no_reuse():
    """Test Showing Parallel overhead by not Reusing a pool of workers"""
    accumulator = 0.
    n_iter = 0
    while accumulator < 1000:
        results = Parallel(n_jobs=2)(delayed(sqrt)(accumulator + i ** 2) for i in range(5))
        accumulator += sum(results)  # synchronization barrier
        n_iter += 1


# In[8]:

get_ipython().magic(u'timeit test_reuse()')


# In[9]:

get_ipython().magic(u'timeit test_no_reuse()')


# # Generators:
# Similar to comprehension lists but is effeicent with memory. When you create a comprehension list you need to store it in memory. This can be a problem if you use very large arrays.
# 
# The generator only creates one value at a time and then when it has used that value it forgets about it. Thus saving memory. As a result they can be used for iteration but only once.
# You create a generator by using normal brackets "()" instead of square brackets "[]".

# In[12]:

List = [x ** 2 for x in range(10) if (x%3) is 0]
print(List)
for val in List:
    print(val)


# In[13]:

gen = (x ** 2 for x in range(10) if (x%3) is 0)
print(gen)
for val in gen:
    print(val)


# In[14]:

print("Another-iteration") 
print(List)
for val in List:
    print(val)
    
# Re-iteration of generator does not return any more values
print(gen)    
for val in gen:
    print(val)


# For large arrays it may be more effeicent to create a generator function more than once instead of have a large list saved in memory.

# ## File Processing thought example
# 
# Say you had many files that needed information out of or some processing/transformation.
# Say you wanted to extract some information from each of the files e.g. the time, coordinates, some other header information
# Or you wanted to normalize spectra (spectra.fits) and save the result to a new file (spectra_normalised.fits)
# 
# If all the files (input and output) are independant and your processing automatic then you would probably loop over the files. This should be able to be parallelized.
# 
# ### Warning nested parallel processes are probably not a good idea. 

# In[ ]:

#1/2 Code 1/2 Psudocode for many file example:
filenames = ["file1.txt", "file2.txt", ...., "fileN.txt"]
#Serial example
for fname in filenames:
    # Open file and load in data
    with open(fname,"r") as f:
        # Read in data
        data = f.readlines()
    # Do task
    ans = calculations(data)
    
    #Exctract some information and/or # Save to a file
    with open(savefile, "w") as g:
        # Output to file
        g.write(ans)
        
    return ans


# Turn the code inside the loop into its own function
def file_processing(filename, *args):
     # Open file and load in data
    with open(fname,"r") as f:
        # Read in data
        data = f.readlines()
    # Do task
    ans = calculations(data)
    
    #Exctract some information and/or # Save to a file
    with open(savefile, "w") as g:
        # Output to file
        g.write(ans)
        
    return ans


# Serial example with function  
for fname in filenames:
    file_processing(filename, *args)
# or as comprehension list
[file_processing(fname, *args) for fname in filenames]


# Parallel with joblib.
Parallel(n_jobs=2)(delayed(file_processing)(fname, *args) for fname in filenames)

# If you need to then you can write code to extract the results from all the separate savefiles


# # Convolution Example
# Convolution without loss of information from interpolation because we don't interpolate to equidistant wavelenght points. 
# 
# Adapted from code from Pedro Figueira

# First off define some functions needed

# In[21]:

import matplotlib.pyplot as plt

def wav_selector(wav, flux, wav_min, wav_max):
    """
    function that returns wavelength and flux withn a giving range
    """    
    wav_sel = np.array([value for value in wav if(wav_min < value < wav_max)], dtype="float64")
    flux_sel = np.array([value[1] for value in zip(wav,flux) if(wav_min < value[0] < wav_max)], dtype="float64")
    
    return [wav_sel, flux_sel]


def unitary_Gauss(x, center, FWHM):
    """
    Gaussian_function of area=1

    p[0] = A;
    p[1] = mean;
    p[2] = FWHM;
    """
    
    sigma = np.abs(FWHM) /( 2 * np.sqrt(2 * np.log(2)) );
    Amp = 1.0 / (sigma*np.sqrt(2*np.pi))
    tau = -((x - center)**2) / (2*(sigma**2))
    result = Amp * np.exp( tau );
    
    return result

def chip_selector(wav, flux, chip):
    chip = str(chip)
    if(chip in ["ALL", "all", "","0"]):
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT1"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END4"])   # Wavelength end on detector [nm]
        #return [wav, flux]
    elif(chip == "1"):
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT1"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END1"])   # Wavelength end on detector [nm]
    elif(chip == "2"):
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT2"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END2"])   # Wavelength end on detector [nm]
    elif(chip == "3"):   
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT3"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END3"])   # Wavelength end on detector [nm]
    elif(chip == "4"):   
        chipmin = float(hdr1["HIERARCH ESO INS WLEN STRT4"])  # Wavelength start on detector [nm]
        chipmax = float(hdr1["HIERARCH ESO INS WLEN END4"])   # Wavelength end on detector [nm]
    elif(chip == "Joblib_small"):   
        chipmin = float(2118)  # Wavelength start on detector [nm]
        chipmax = float(2119)  # Wavelength end on detector [nm]
    elif(chip == "Joblib_large"):   
        chipmin = float(2149)  # Wavelength start on detector [nm]
        chipmax = float(2157)  # Wavelength end on detector [nm]
    else:
        print("Unrecognized chip tag.")
        exit()
    
    #select values form the chip  
    wav_chip, flux_chip = wav_selector(wav, flux, chipmin, chipmax)
    
    return [wav_chip, flux_chip]



# ### Serial version of convolution
# The computationally heavy part is the for loop over each wavelenght value

# In[22]:


def convolution_serial(wav, flux, chip, R, FWHM_lim=5.0, plot=True):
    """Convolution code adapted from pedros code"""
    
    wav_chip, flux_chip = chip_selector(wav, flux, chip)
    #we need to calculate the FWHM at this value in order to set the starting point for the convolution
 
    FWHM_min = wav_chip[0]/R    #FWHM at the extremes of vector
    FWHM_max = wav_chip[-1]/R       
    
    
    #wide wavelength bin for the resolution_convolution
    wav_extended, flux_extended = wav_selector(wav, flux, wav_chip[0]-FWHM_lim*FWHM_min, wav_chip[-1]+FWHM_lim*FWHM_max) 
    wav_extended = np.array(wav_extended, dtype="float64")
    flux_extended = np.array(flux_extended, dtype="float64")
    
    print("Starting the Resolution convolution...")
    
    flux_conv_res = []
    counter = 0    
    for wav in wav_chip:
        # select all values such that they are within the FWHM limits
        FWHM = wav/R
        #print("FWHM of {0} calculated for wavelength {1}".format(FWHM, wav))
        indexes = [ i for i in range(len(wav_extended)) if ((wav - FWHM_lim*FWHM) < wav_extended[i] < (wav + FWHM_lim*FWHM))]
        flux_2convolve = flux_extended[indexes[0]:indexes[-1]+1]
        IP = unitary_Gauss(wav_extended[indexes[0]:indexes[-1]+1], wav, FWHM)
        flux_conv_res.append(np.sum(IP*flux_2convolve))
        if(len(flux_conv_res)%(len(wav_chip)//100 ) == 0):
            counter = counter+1
            print("Resolution Convolution at {}%%...".format(counter))
    flux_conv_res = np.array(flux_conv_res, dtype="float64")
    print("Done.\n")
    
    if(plot):
        fig=plt.figure(1)
        plt.xlabel(r"wavelength [ $\mu$m ])")
        plt.ylabel(r"flux [counts] ")
        plt.plot(wav_chip, flux_chip/np.max(flux_chip), color ='k', linestyle="-", label="Original spectra")
        plt.plot(wav_chip, flux_conv_res/np.max(flux_conv_res), color ='b', linestyle="-", label="Spectrum observed at and R=%d ." % (R))
        plt.legend(loc='best')
        plt.show() 
    return wav_chip, flux_conv_res


# ### Parallel version of convolution

# In[23]:

# Function around bottleneck
def convolve(wav, R, wav_extended, flux_extended, FWHM_lim):
        # select all values such that they are within the FWHM limits
        FWHM = wav/R
        indexes = [ i for i in range(len(wav_extended)) if ((wav - FWHM_lim*FWHM) < wav_extended[i] < (wav + FWHM_lim*FWHM))]
        flux_2convolve = flux_extended[indexes[0]:indexes[-1]+1]
        IP = unitary_Gauss(wav_extended[indexes[0]:indexes[-1]+1], wav, FWHM)
        val = np.sum(IP*flux_2convolve) 
        unitary_val = np.sum(IP*np.ones_like(flux_2convolve))  # Effect of convolution onUnitary. For changing number of points
        return val/unitary_val
    
def convolution_parallel(wav, flux, chip, R, FWHM_lim=5.0, n_jobs=-1, verbose=5):
    """Convolution code adapted from pedros code"""
    
    wav_chip, flux_chip = chip_selector(wav, flux, chip)
    #we need to calculate the FWHM at this value in order to set the starting point for the convolution
    
    #print(wav_chip)
    #print(flux_chip)
    FWHM_min = wav_chip[0]/R    #FWHM at the extremes of vector
    FWHM_max = wav_chip[-1]/R       
    
    #wide wavelength bin for the resolution_convolution
    wav_extended, flux_extended = wav_selector(wav, flux, wav_chip[0]-FWHM_lim*FWHM_min, wav_chip[-1]+FWHM_lim*FWHM_max) 
    wav_extended = np.array(wav_extended, dtype="float64")
    flux_extended = np.array(flux_extended, dtype="float64")
    
    print("Starting the Parallel Resolution convolution...")
    
    parallel_result = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(convolve)(wav,R,wav_extended, flux_extended,FWHM_lim) for wav in wav_chip)
    flux_conv_res = np.array(parallel_result, dtype="float64")
    print("Done.\n")
    

    return wav_chip, flux_conv_res 



# In[27]:

# Load data
import numpy as np
#wl, flux = np.loadtxt("Joblib_tapas.txt")  # 2117-2120 nm
wl, flux = np.loadtxt("Joblib_tapas_large.txt")  # 2145-2160 nm


# Time a serial convolution

# In[28]:

import time
import datetime
start = time.time()

# "Joblib_small"  # "Joblib_large"
x, y = convolution_serial(wl, flux, "Joblib_large", 50000, FWHM_lim=5.0, plot=False)
  
done = time.time()
elapsed = done - start
print("Convolution time = ", elapsed)


# Time a parallel convolution

# In[29]:

start = time.time()
# "Joblib_small", "Joblib_large"

x_par, y_par = convolution_parallel(wl, flux, "Joblib_large", 50000, FWHM_lim=5.0)
  
done = time.time()
elapsed = done - start
print("Convolution time = ", elapsed)


# # Logging
# Traceback example, note how the line of the error is indicated as well as the values of the parameter passed to the function that triggered the exception, even though the traceback happens in the child process.
# 

# In[31]:

from heapq import nlargest
from joblib import Parallel, delayed
Parallel(n_jobs=3)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) 


# # Joblibs Other tools
# 
# ## Memory
# Example from Joblib documentation showing the caching of input and outputs of the function sqaure().
# 
# When it is called with the same parameters again it jsut returns the result without recomputation.

# In[32]:

from joblib import Memory
mem = Memory(cachedir='/tmp/joblib')
import numpy as np
a = np.vander(np.arange(101)).astype(np.float)
b = np.vander(np.arange(5)).astype(np.float)
square = mem.cache(np.square)


# In[33]:

c = square(a) 


# In[34]:

d = square(b)


# In[35]:

e = square(a) # Does not recomute square(a)


# In[36]:

f = square(b) # Does not recomute square(b)


# Timing these calls to square shows that the second call of the function with the same inputs give a much faster result.

# ## Persistance
# joblib.dump() and joblib.load() provide a replacement for pickle to work efficiently on Python objects containing large data, in particular large numpy arrays.
# 
# Filename is important here, .pkl will make a pickle like persistance
# where as .mmap with make a memory map location for parallel process shared access.

# In[41]:

from tempfile import mkdtemp
savedir = mkdtemp()
import os
#filename = os.path.join(savedir, 'test.pkl')      # Pickle version  
filename = os.path.join(savedir, 'test.mmap')      # Memmap version


# In[42]:

#Then we create an object to be persisted:
import numpy as np
#to_persist = [('a', [1, 2, 3]), ('b', np.arange(10))]
to_persist = np.ones(int(1e6))


# In[43]:

#which we save into savedir:
import joblib
joblib.dump(to_persist, filename)  


# In[44]:

# We can then load the object from the file:
#joblib.load(filename)
pointer = joblib.load(filename, mmap_mode='r+')


# In[ ]:



