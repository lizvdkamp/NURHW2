import numpy as np
import matplotlib.pyplot as plt
import timeit

#Question 2

#2a

#Inputting the functions for root finding, Secant, False position, and Newton Rapson
def Secant(func, a, b, acc, maxit):
    """A function that finds the root of a function func between [a,b] based on the Secant method. Terminates when the accuracy < acc or the amount of iterations exceeds maxit."""

    #check for roots
    if func(a)*func(b) > 0:
        print("No root in between range")
        return None
    
    #initialize values for the amount of iterations, the size of the interval, c, and the error
    it = 0
    intsize = b-a
    c = b+(b-a)/(func(a)-func(b)) * func(b)
    err = np.abs(func((c)))

    #iterate
    while (np.abs(intsize) > acc and err > acc) and it < maxit:
        c = b+(b-a)/(func(a)-func(b)) * func(b)
        err = np.abs(func(c))
        
        a = b
        b = c
        it += 1
        intsize = np.abs(b-a)
            
        #print(a,b,intsize,it)
    
    root = c
    #Print the amount of iterations
    print("Done, steps taken = ", it)
    return root 

def Falsepos(func, a, b, acc, maxit):
    """Finds the root of a function func between [a,b] based on the False Position method. Terminates when the accuracy < acc or the amount of iterations exceeds maxit."""

    #Check for a root
    if func(a)*func(b) > 0:
        print("No root in between range")
        return None
    
    #initialize values for the amount of iterations, the size of the interval, c, and the error
    it = 0
    intsize = b-a
    c = b+(b-a)/(func(a)-func(b)) * func(b)
    err = np.abs(func((c)))

    #iterate
    while (intsize > acc and err > acc) and it < maxit:
        c = b+(b-a)/(func(a)-func(b)) * func(b)
        err = np.abs(func(c))
        
        #print(intsize, err, acc, (intsize > acc and err > acc))
        #print(c, (b-a)/(func(a)-func(b)) * func(b))
        
        #make sure that we have a bracket
        if func(a)*func(c) < 0:
            #print(a,c,func(a)*func(c))
            a = a
            b = c
            intsize = np.abs(b-a)
            it +=1
        
        elif func(b)*func(c) < 0:
            #print(c,b,func(b)*func(c))
            a = b
            b = c
            intsize = np.abs(b-a)
            it +=1

        #If there is no root, we terminate
        else:
            print("Whoops we lost the root")
            return None
            
        #print(a,b,intsize,it)
    
    root = c
    print("Done, steps taken:", it)
    return root 

def NewRap(func, deriv, a, acc, maxit):
    """Finds the root of function func with analytical derivative deriv around the point a using the Newton Rapson method. Terminates when the accuracy < acc or the amount of iterations exceeds maxit."""
    
    #initialize values for the amount of iterations, c, and the error
    it = 0

    c = a - func(a)/deriv(a)
    err = np.abs(func(c))
    
    #iterate
    while err > acc and it < maxit:
        c = a - func(a)/deriv(a)
        err = np.abs(func(c))
        
        #print(a, c, err, it)
        
        a = c
        it+=1
    
    root = c
    print("Done, steps taken:", it)
    return root


#Taking from heatingcooling.py
k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s

# here no need for nH nor ne as they cancel out
# I added default values as the values from the problem
def equilibrium1(T,Z=0.015,Tc=10**4,psi=0.929):
    """Returns the heating produced by photoionization minus the radiative recombination divided by the recombination coefficient and the number density of electrons and hydrogen."""
    return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k

def dereq1(T,Z=0.015):
    """Returns the derivative of the equilibrium1 function."""
    return -(0.684 - 0.0416 * np.log(T/(1e4 * Z*Z))- 0.0416)*k

#Check the times for False Position & Newton Rapson
start = timeit.default_timer()

eqT = Falsepos(equilibrium1, 1, 10**7, 10**(-18), 30)

end = timeit.default_timer()-start
print("Time taken to calculate the equilibrium Temperature with False position:", end, "s")
print("Root:", eqT, "Function value:", equilibrium1(eqT))


start2 = timeit.default_timer()

eqT2 = NewRap(equilibrium1, dereq1, 10**(3.5), 10**(-18), 30)

end2 = timeit.default_timer()-start2
print("Time taken to calculate the equilibrium Temperature with Newton Rapson:", end2, "s")
print("Root:", eqT2, "Function value:", equilibrium1(eqT2))

# Save a text file with the times
np.savetxt('Timesoutput2a.txt',np.transpose([end,end2]))

#The Newton Rapson method converges fastest

#A plot to show the roots and the function
Ts = np.linspace(1,10**7,10**6)
plt.plot(Ts, equilibrium1(Ts), label='curve')
plt.plot(Ts, np.zeros(len(Ts)), label='zero line')
plt.scatter(eqT,equilibrium1(eqT), label='root False Pos')
plt.scatter(eqT2,equilibrium1(eqT2), label='root NR')
plt.xscale('log')
plt.ylim(-5*10**(-19),8*10**(-19))
plt.xlim(eqT-0.1,eqT2+0.1)
plt.legend()
plt.savefig('Temperatureplot2a.png')
plt.close()


#2b

#Define a basic root finding function that uses bisection

def basicroot(func, a, b, acc, maxit):
    """Finds the root of a function func between [a,b] using bisection. Terminates when the accuracy < acc or the amount of iterations exceeds maxit."""
    
    #check for a root
    if func(a)*func(b) > 0:
        print("No root in between range")
        return None
    
    #initialize values for the amount of iterations, the size of the interval, c, and the error
    it = 0
    intsize = np.abs(b-a)
    c = (a+b)/2
    err = np.abs(func(c))
    #print(intsize, acc, it, maxit)
    while intsize > acc and err > acc and it < maxit:
        c = (a+b)/2
        err = np.abs(func(c))
        
        #check which interval forms a bracket
        if func(a)*func(c) <= 0:
            #print(a,c,func(a)*func(c))
            a = a
            b = c
            intsize = np.abs(b-a)
            it +=1
        
        elif func(b)*func(c) <= 0:
            #print(c,b,func(b)*func(c))
            a = c
            b = b
            intsize = np.abs(b-a)
            it +=1
            
        else:
            print("Whoops we lost the root")
            return None
            
        #print(a,b,intsize,it)
        
    #taking the root in the middle of the interval
    root = (a+b)/2
    print("Done, steps taken:", it)
    return root

#defining ne
ne = 1

def equilibrium2(T,Z=0.015,Tc=10**4,psi=0.929, A=5*10**(-10),xi=10**(-15)):
    """Returns the heating produced by photoionization, cosmic rays and MHD waves minus radiative recombination and free-free emission."""
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)

def dereq2(T,Z=0.015):
    """Returns the derivative of the equilibrium2 function."""
    return (-(0.684 - 0.0416 * np.log(T/(1e4 * Z*Z))- 0.0416) - .54 * 1.37* ( T/1e4 )**.37)*k*nH*aB + 8.9e-26 * (1/1e4)



#Defining some variables
nes = np.array([10**(-4),1,10**4])
Tstart = 1
Tend = 10**15
targacc = 10**(-18)
maxit = 60

#An empty list to save the times it takes to calculate this
times = []

for i in range(len(nes)):
	nH = nes[i]
	print("nH = ", nH)

	if nH > 10**(-4):
		start = timeit.default_timer()

		eqT = Falsepos(equilibrium2, Tstart, Tend, targacc, maxit)

		end = timeit.default_timer()-start
		print("Time taken to calculate the equilibrium Temperature with False position:", end, "s")
		print("Root:", eqT, "Function value:", equilibrium2(eqT))
		times.append(end)


	else:
		start2 = timeit.default_timer()

		eqT2 = basicroot(equilibrium2, Tstart, Tend, targacc, maxit)

		end2 = timeit.default_timer()-start2
		print("Time taken to calculate the equilibrium Temperature with Bisection:", end2, "s")
		print("Root:", eqT2, "Function value:", equilibrium2(eqT2))
		times.append(end2)

#print(times)
# Save a text file
np.savetxt('Timesoutput2b.txt',np.transpose([times[0],times[1],times[2]]))







