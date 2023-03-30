import numpy as np
import matplotlib.pyplot as plt

#Question 1

#1a

#Defining the needed functions for integration
def trapezoid(a,b,N,func):
	"""Takes a function 'func' and calculates the trapezoidal area underneath the function for on the interval [a,b] with stepsize (b-a)/N"""
	#step size
	h = (b-a)/N

	xs = np.linspace(a,b,N) #x
	fxs = func(xs)		#f(x)

	#trapezoidal area
	trpzd = h*(fxs[0]*0.5 + np.sum(fxs[1:N-1]) + fxs[N-1]*0.5)

	return trpzd

#Function for Romberg integration
def Romberg(a,b,N,m,func):
	"""Calculates the integral of a function 'func' over the interval [a,b] by iterating m times over a trapezoidal integration with initial stepsize (b-a)/N"""

	h = (b-a)/N	#initial stepsize
	r = np.zeros(m)
	#Initial guess
	r[0] = trapezoid(a,b,N,func)
	Np = N

	#First loop where we iterate over different stepsizes
	for i in range(1,m):
		r[i] = 0
		delta = h
		h = h*0.5
		x = a+h
        
		for n in range(Np):
			r[i] += func(x)
			x += delta
		    
		#Initial guess
		r[i] = 0.5*(r[i-1]+delta*r[i])
		Np *= 2
    
    
	Np = 1
	#Improving by iterating m times
	for i in range(1,m):
		Np *= 4
		for j in range(0, m-i):
			r[j] = (Np*r[j+1]-r[j])/(Np-1)
            
	return r[0]


#defining a function for n(x) and defining the constants

A = 1 #to be computed
a = 2.4
b = 0.25
c = 1.6

x_max = 5

Nsat = 100

#n(x) does not depend on theta or phi, so dV reduces to 4*pi*x^2*dx

def n2x(x,a=2.4,b=0.25,c=1.6,Nsat=100,A=1):
	"""Returns 4*pi*n(x)*x**2"""
	return 4*np.pi*A*Nsat*((x**(a-1)/b**(a-3))*np.exp(-(x/b)**c))

#Since we have A = 1 here, the integral will calculate <Nsat>/A, we will obtain A by taking the result of the integral and then dividing by 100

A_100 = Romberg(0,x_max,20,8,n2x)
print("The Romberg integration with N = 20 and m = 8 results in <Nsat>/A = ", A_100)
A_int = 100/A_100
print("The result for A = ", A_int)

# Save a text file
np.savetxt('Integrationoutput.txt',[A_int])


#-------
#1b

#Assuming that N(x)dx = x**2*n(x)dx, such that the integral from 0 to x_max over N(x)dx/<Nsat> = 1, just like a probability distribution should do

def Nxdx(x,a=2.4,b=0.25,c=1.6,Nsat=100,A=A_int):
	"""Returns N(x)dx / <Nsat> = 4*pi*x**2*n(x)dx / <Nsat>"""
	return 4*np.pi*A*((x**(a-1)/b**(a-3))*np.exp(-(x/b)**c))

#Since n(x) is not invertible or easily integrated analytically, I will use Rejection sampling to sample the distribution

#Defining the function for generating random numbers

def lcgbit(i0, size, a=1664525, m=2**32, c=1013904223, norm=True, a1=21,a2=25,a3=4):
    """Takes a seed i0 and generates 'size' amount of random numbers, by combining 64-bit XOR and an LCG. If norm=True, then it returns a random uniform distribution between 0 and 1."""

    #Initialize arrays to fill
    randnrs = np.zeros(size, dtype=int)
    xors = np.zeros(size+1, dtype=int)
    xors[0] = int(i0)
    #print("seed = ", i0)
    
    #First do the XOR shift
    for i in range(1,size+1):
        x = xors[i-1]
        x1 = x ^ (x >> a1)
        x2 = x1 ^ (x1 << a2)
        x3 = x2 ^ (x2 >> a3)
        xors[i] = x3
        #print("x, x1, x2, x3", x,x1,x2,x3)

    #Here we do the LCG
    for j in range(0,size):
        ii = (a*xors[j+1]+c)%m
        randnrs[j] = ii	#add the random number to the array
        #print("ii", ii)
        
    if norm:
        #Return a normalized array
        return randnrs/np.amax(randnrs)
    else:
        return randnrs


def Rejsamp(N, a, b, px):
	"""Takes a range [a,b] and a probability function p(x) and returns N points that follow the distribution"""
	x_sample = np.zeros(N)

	#Obtain the maximum value of the probability distribution
	xs = a+(b-a)*lcgbit(3,size=N) #U(a,b)
	pxs = px(xs)
	maxp = np.amax(pxs)
	#print(maxp)

	#index j and seed generator k
	j = 0
	k = 0
	
	while j < N:
		rands = lcgbit(k+1,size=100) #U(0,1)
		x = a+(b-a)*rands[0] #1 number from U(a,b)
		y = maxp*rands[1] #1 number from U(0,max(p(x)))
		#print("x, y =", x,y)
		
		if y <= px(x):
			#accept
			#print("Accept", j)
			x_sample[j] = x
			j += 1
			k += 1
		else:
			#reject, try a different seed for the random number generator
			k += 1

	return x_sample

#getting the sample of x values
x_min = 10**(-4)
Number = 10**4
x_distr = Rejsamp(Number,x_min,x_max,Nxdx)

#taking a uniform sample of x values to plot Nxdx
xes = np.linspace(x_min,x_max,Number)

#bins, equally spaced in log space
bins = np.logspace(np.log10(x_min),np.log10(x_max), 20)

#Plotting
plt.loglog(xes, Nxdx(xes), label="N(x)", color='k')
#By taking density = True we divide the bins by their width and divide by the number of counts
plt.hist(x_distr, bins=bins, density=True, label="sampled x", log=True, rwidth=0.95, color='crimson')
plt.ylabel(r"$p(x)$d$x = N(x)$d$x / <N_{sat}>$")
plt.xlabel(r"$x = r/r_{vir}$")
plt.legend()
plt.savefig('Probabilityplot.png')
plt.close()


#-------

#1c

#Defining the functions for row swaps and Quicksort

def SwapRowVec(x,i,j):
	"""Swaps row i and j of vector x"""

	B = np.copy(x).astype('float64')

	#Swap
	save = B[i].copy()
	B[i]= B[j]
	B[j] = save

	return B

def Quicksort_part(arr, ind_low, ind_high, indxsave, indxs):
	"""Function which sorts the pivot and then breaks the array into partial arrays around the pivot and sorts those iteratively"""
	#get the right part of the array
	a = arr[ind_low:ind_high+1]
	Ntot = len(arr)
	N = len(a)
	middle = int(N*0.5)

	if N < 3:
		return arr, indxs
    
	#save the pivot
	x_piv = a[middle]

	#save the outer indices from the partial array
	i = ind_low
	j = ind_high
	#save the outer indices from the total array
	i_2 = 0
	j_2 = N-1
	for k in range(0,N):
		#print("i,j", i,j, arr[i], arr[j], x_piv)
		if i >= j:
		    #stop when the index that goes from the left exceeds the one from the right
		    break
		#if the number is larger than the pivot, do nothing for now
		if arr[i] >= x_piv:
			pass
		#if it's smaller, its on the right side of the pivot, so we go up one
		else:
			i += 1
			i_2 += 1
		#check if the numbers on the right side of the pivot are smaller, otherwise lower the indices
		if arr[j] <= x_piv:
			pass
		else:
			j -= 1
			j_2 -= 1
		#if we found two on the wrong side, switch (both on the total and the partial array)
		if arr[i] >= x_piv and arr[j] <= x_piv:
			#print("i,j", i,j, arr[i], arr[j], x_piv)
			#print("swap")
			arr = SwapRowVec(arr,i,j)
			a = SwapRowVec(a,i_2,j_2)
		#print(arr)
		if indxsave:
			indxs = SwapRowVec(indxs,i,j)
                        
	#Sorting the parts around the pivot

	low = a[0]
	high = a[N-1]

	indxes = np.arange(0,Ntot,1, dtype=int)
	indxes2 = np.arange(0,N,1, dtype=int)

	indx_piv = (indxes[arr == x_piv])[0]
	indx_piv2 = (indxes2[a == x_piv])[0]
	in_low = (indxes[arr == low])[0]
	in_high = (indxes[arr == high])[0]
	#print("indx", indx_piv)
	#print("indx2", indx_piv2, N)
	#print("low, high ind", in_low, in_high)

	if indx_piv2 <= 1:
		#print("small",indx_piv+1, in_high)
		arr, indxs = Quicksort_part(arr, indx_piv+1, in_high, indxsave, indxs)
	if indx_piv2 >= N-2:
		#print("large",in_low, indx_piv)
		arr, indxs = Quicksort_part(arr, in_low, indx_piv, indxsave, indxs)
	else:
		#print("else")
		arr, indxs = Quicksort_part(arr, in_low, indx_piv, indxsave, indxs)
		arr, indxs = Quicksort_part(arr, indx_piv+1, in_high, indxsave, indxs)
        

	return arr, indxs

  
def Quicksort(arr, indxsave=False):
	"""Main function for Quicksort, identifies the pivot"""

	#copying the array and saving the indices & middle incdex
	a = arr.copy()
	N = len(arr)
	middle = int(N*0.5)
	#make an indxs array in case we want to shuffle an array
	indxes = np.arange(0,N,1, dtype=int)


	#print(a)

	#Take the first middle and last part of the array and put those in order
	fml = np.array([a[0], a[middle], a[N-1]]).copy()
	fml_ind = np.array([indxes[0], indxes[middle], indxes[N-1]]).copy()

	a[0] = np.amin(fml)
	indxes[0] = fml_ind[fml==np.amin(fml)][0]

	a[N-1] = np.amax(fml)
	indxes[N-1] = fml_ind[fml==np.amax(fml)][0]


	if fml[(fml > a[0]) & (fml < a[N-1])].size == 0:
		#this means that there are 2 equal numbers
		pass
	else:
		a[middle] = fml[(fml > a[0]) & (fml < a[N-1])]
		indxes[middle] = fml_ind[fml==fml[(fml > a[0]) & (fml < a[N-1])]][0]
    
	#print(a)

	#Saving the pivot
	x_piv = a[middle]

	#first and last numbers of the array
	low = a[0]
	high = a[N-1]

	in_low = 0
	in_high = N-1

	#calling the partial sorting array
	a, indxes = Quicksort_part(a, in_low, in_high, indxsave, indxes)        

	if indxsave==True:
		return indxes.astype(int)
    
	return a


#I will use Quicksort to create shuffle the x_sample and pick 100 random samples with equal probability
#shuffling indices by sorting a random array
randarr = lcgbit(5, size=Number)

#random indices
randinx = Quicksort(randarr, indxsave=True)

#random 100 galaxies, no rejection, no galaxy twice
x_shuffle = x_distr[randinx][0:100]

#Now sorting this
x_100sort = Quicksort(x_shuffle, indxsave=False)
#print(x_100sort[95:])
#The amount of galaxies within a radius here is equal to 0-100 at the sorted radii, and it is 0 at x_min
amount = np.arange(0,x_100sort.size+1,1)

#Making sure that we go from x_min to x_max, so I insert those into the sorted x array, and I insert 100 at x_max because we have 100 galaxies
x_100sort = np.insert(x_100sort, 0, x_min)
x_100sort = np.insert(x_100sort, x_100sort.size, x_max)
amount = np.append(amount, 100)

#Plotting

plt.plot(x_100sort,amount)
plt.xlim(x_min,x_max)
plt.xscale('log')
plt.ylabel("Number of galaxies within a radius x")
plt.xlabel(r"$x = r/r_{vir}$")
plt.savefig('Numberplot.png')
plt.close()

#print(x_shuffle, x_shuffle.size)

#-------

#1d

#Defining the functions to calculate the derivative numerically
def centraldif(h,x,func):
	"""Calculates the derivative with the central difference method for stepsize h"""
	dfdx = (func(x+h)-func(x-h))/(2*h)

	return dfdx

def Ridder(h,x,func,error,analder,m=100):
	"""Uses Ridder's method to calculate the numerical derivative, similar to Romberg integration, by iterating over different solutions. The analytical derivative here is only used to calculate the error and to make sure that it stays below the target error. h is the stepsize, error is the target error, m is the amount of iterations."""
	
	r = np.zeros((m,x.size))
	#Initial guess
	r[0,:] = centraldif(h,x,func)
	d = 2 
	d_inv = 1/d

	#First loop
	for i in range(1,m):
		delta = h #stepsize
		h = h*d_inv #halving new stepsize
		#Calculating new solution with half the stepsize
		r[i,:] = centraldif(h,x,func)
    
	Np = 1
	#Improving
	for i in range(1,m):
		Np *= d**2

		#Keep track of the error between our best guess and the analytical derivative to make sure that it doesn't grow
		errsolprev = np.mean(np.abs(r[0,:]-analder(x))) #Error on previous guess
		solprev = r[0,:].copy()	#Saving the previous guess in case the error grows
		for j in range(0, m-i):
			r[j,:] = (Np*r[j+1,:]-r[j,:])/(Np-1)
        
		errsol = np.mean(np.abs(r[0,:]-analder(x))) #Error on current guess
		#Check if the error grows
		if errsolprev < errsol:
			print("error grows", errsol)
			return solprev
		if errsol < error:
			#print(i, errsol)
			return r[0,:]

	return r[0,:]

#Fuctions
def nx(x,a=2.4,b=0.25,c=1.6,Nsat=100,A=A_int):
	"""Returns n(x)*x**2"""
	return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

#Analytical derivative
def dndx(x,a=2.4,b=0.25,c=1.6,Nsat=100,A=A_int):
	"""Returns the analytical derivative of n(x), dn/dx"""
	return A*Nsat*(x**(a-4)/b**(a-3))*np.exp(-(x/b)**c) * (a-3-c*(x/b)**c)

#The function is made to take arrays for x, so I make [1,1,1]
xs = np.ones(3)

#analytic result
analder = dndx(xs)[0]
#numerical
numder = Ridder(0.1,xs,nx,10**(-10),dndx,m=100)[0]

print("Result analytic derivative", analder)
print("Result numerical derivate", numder)
print("Absolute difference", np.abs(numder-analder))
# Save a text file
np.savetxt('Derivativeoutput.txt',np.transpose([analder,numder,np.abs(numder-analder)]))








