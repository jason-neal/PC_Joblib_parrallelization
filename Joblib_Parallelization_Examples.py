
# coding: utf-8

# In[ ]:

# Embarisingly Parallel for loops using Joblib
Example from documentation


# In[102]:

from math import sqrt
[sqrt(i ** 2) for i in range(10)]


# In[101]:

from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))


# The progress meter: the higher the value of verbose, the more messages:
# If verbose > 50 then message a for every task is returned.

# In[97]:

from time import sleep
from joblib import Parallel, delayed
# n_jobs=1 turns off the parallel code for debuging.
r = Parallel(n_jobs=1, verbose=1)(delayed(sleep)(.1) for _ in range(100)) 


# In[98]:

r = Parallel(n_jobs=2, verbose=5)(delayed(sleep)(.1) for _ in range(100)) 


# In[99]:

r = Parallel(n_jobs=-1, verbose=10)(delayed(sleep)(.1) for _ in range(100)) 


# Running the same code on the simple sleep function shows the effect of increasing the number of seperate jobs/processes.

# ## Reusing a pool of workers

# In[208]:

def test_reuse():
    """Test Reusing a pool of workers """
    with Parallel(n_jobs=2) as parallel:
        accumulator = 0.
        n_iter = 0
        while accumulator < 1000:
            results = parallel(delayed(sqrt)(accumulator + i ** 2) for i in range(5))
            accumulator += sum(results)  # synchronization barrier
            n_iter += 1


# In[209]:

def test_no_reuse():
    """Test Showing Parallel overhead by not Reusing a pool of workers"""
    accumulator = 0.
    n_iter = 0
    while accumulator < 1000:
        results = Parallel(n_jobs=2)(delayed(sqrt)(accumulator + i ** 2) for i in range(5))
        accumulator += sum(results)  # synchronization barrier
        n_iter += 1


# In[ ]:

get_ipython().magic('timeit test_reuse()')


# In[ ]:

get_ipython().magic('timeit test_no_reuse()')


# # Generators:
# Similar to comprehension lists but is effeicent with memory. When you create a comprehension list you need to store it in memory. This can be a problem if you use very large arrays.
# 
# The generator only creates one value at a time and then when it has used that value it forgets about it. Thus saving memory. As a result they can be used for iteration but only once.
# You create a generator by using normal brackets "()" instead of square brackets "[]".

# In[ ]:

List = [x ** 2 for x in range(10) if (x%3) is 0]
print(List)
for val in List:
    print(val)


# In[ ]:

gen = (x ** 2 for x in range(10) if (x%3) is 0)
print(gen)
for val in gen:
    print(val)


# In[ ]:

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
# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Joblibs Other tools
# 
# ## Memory
# Example from Joblib documentation showing the caching of input and outputs of the function sqaure().
# 
# When it is called with the same parameters again it jsut returns the result without recomputation.

# In[ ]:

from joblib import Memory
mem = Memory(cachedir='/tmp/joblib')
import numpy as np
a = np.vander(np.arange(10001)).astype(np.float)
b = np.vander(np.arange(5)).astype(np.float)
square = mem.cache(np.square)


# In[ ]:

get_ipython().magic('time c = square(a)')


# In[ ]:

get_ipython().magic('time d = square(b)')


# In[ ]:

get_ipython().magic('time e = square(a) # Does not recomute square(a)')


# In[ ]:

get_ipython().magic('time f = square(b) # Does not recomute square(b)')


# Timing these calls to square shows that the second call of the function with the same inputs give a much faster result.

# ## Persistance
# joblib.dump() and joblib.load() provide a replacement for pickle to work efficiently on Python objects containing large data, in particular large numpy arrays.
# 
# Filename is important here, .pkl will make a pickle like persistance
# where as .mmap with make a memory map location for parallel process shared access.

# In[ ]:

from tempfile import mkdtemp
savedir = mkdtemp()
import os
filename = os.path.join(savedir, 'test.pkl')
#filename = os.path.join(savedir, 'test.mmap')


# In[ ]:

#Then we create an object to be persisted:
import numpy as np
to_persist = [('a', [1, 2, 3]), ('b', np.arange(10))]
#to_persist = np.ones(int(1e6))


# In[ ]:

#which we save into savedir:
import joblib
joblib.dump(to_persist, filename)  


# In[ ]:

# We can then load the object from the file:
joblib.load(filename)
#joblib.load(filename, mmap_mode='r+')


# In[ ]:




# In[ ]:




# In[ ]:



