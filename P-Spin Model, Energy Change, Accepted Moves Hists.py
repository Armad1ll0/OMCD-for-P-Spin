# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:27:11 2021

@author: amill
"""

#p-spin model, p=3 (triplets)
import time 
start = time.time()
import numpy as np 
import random as random 
import math as math 
import itertools 
import matplotlib.pyplot as plt 
import multiprocessing 
from collections import Counter 


#size of the lattice 
#NOTE: P spin does not have to be a lattice so we need to change this code so it is not an array and just a list.  
p=3

#number of particles 
N = 27
d0 = 12

def set_up(d0):
    #generating a random list of coords with a gaussian dist N[0, 1]
    variance = math.factorial(p)/(2*(N**(p-1)))
    strd_dev = math.sqrt(variance)
    spins = [np.random.normal(loc = 0.0, scale = 1) for i in range(N)] 
    spins = np.asarray(spins)
    
    #we need to normalize the coords so that the squared sum of them = N so this is what this does. 
    N_inverse = 1/N
    squared_spins = spins**2
    sum_square_spins = np.sum(squared_spins)
    factor = sum_square_spins*N_inverse
    inverse_factor = 1/math.sqrt(factor)
    
    #Note: due to floating point arithmetic in python, the sum of the normalised coords may be slghtly more or less than N 
    original_normalised_spins = spins*inverse_factor 
    
    #now need to create a probability dist of J values to associate with each triplet 
    #this is the gaussian dist that J is taken from 
    variance = math.factorial(p)/(2*(N**(p-1)))
    strd_dev = math.sqrt(variance)
    J_dist = np.random.normal(loc = 0.0, scale = strd_dev)
    
    #listing the possible triplets for the p-spin model 
    triplets = list(itertools.combinations(original_normalised_spins, 3))
    #this print fucntion shows that we have managed to create all possible pairs of triplets  
    J_vals = [np.random.normal(loc = 0.0, scale = strd_dev) for i in range(len(triplets))]
    J_vals = np.asarray(J_vals)
    
    #Now we can calculate the energy of the triplets 
    spins_multiply = [(x*y*z) for x, y, z in triplets]
    spins_multiply = np.asarray(spins_multiply) 
    before_summed = spins_multiply*J_vals*(-1)
    total_energy = np.sum(before_summed)

    
    #need to think about how we perturb the particles, maybe we pick d0 amount of spins to change and perturb them by some amount
    #then renormalize the the coordinates and calculate the new energy from this. 
    energy_vals = [total_energy]
    energy_vals_cont = [total_energy]
    acceptance = 0
    rejection = 0 
    d0 = 12
    
    #need to define a montecarlo step 
    monte_carlo_step = round(N/d0)
    steps = 100
    thousand_monte_carlo_steps = monte_carlo_step*steps
    normalised_spins = original_normalised_spins.copy()
    return thousand_monte_carlo_steps, normalised_spins, total_energy, acceptance, rejection, J_vals, N_inverse, strd_dev, d0

samples = 100

def OMCD(thousand_monte_carlo_steps, normalised_spins, total_energy, acceptance, rejection, J_vals, N_inverse, strd_dev, d0):
    first_energy = total_energy 
    first_spins = normalised_spins
    first_J_vals = J_vals 
    first_N_inverse = N_inverse
    d0_when_energy_change = []
    accepted_energy_change = []
    energy_vals = [total_energy]
    energy_vals_cont = [total_energy]
    count = 0
    total_energy = first_energy
    normalised_spins = first_spins 
    J_vals = first_J_vals
    N_inverse = first_N_inverse
    graph_energy = []
    d0_starting_vals = []
    while d0 > 0: 
    #for i in range(100):
            
            new_normalised_spins = normalised_spins.copy()
            
            #randomly choosing d0 amount of coords to change the spin of  
            changing = []
            for i in range(N):
                changing.append(i)
                
            random.shuffle(changing)
            
            change_these = changing[0:d0]
            
            #perturbing the first few spins of the system 
            for i in change_these:
              new_normalised_spins[i] += np.random.normal(loc = 0.0, scale = strd_dev)
            
            #we have to renormalize them in this part  
            squared_change = new_normalised_spins**2
            sum_square_change = np.sum(squared_change)
            change_factor = sum_square_change*N_inverse
            inverse_change_factor = 1/math.sqrt(change_factor)   
            new_normalised_spins = new_normalised_spins*inverse_change_factor
 
            #now we need to recalulate the energy, accept/reject it and do the process over and over again 
            #doing the same procedure as before 
            #the reason we can do this is because python will always produce a list of triplets in the same order due to its programming. 
            #it does not produce unique triplets randomly, therefore everytime we re assign it, the same spins will be put in the same triplets even when they are perturbed.
            new_triplets = list(itertools.combinations(new_normalised_spins, 3))
            spins_multiply = [(x*y*z) for x, y, z in new_triplets]
            spins_multiply = np.asarray(spins_multiply) 
            before_summed = spins_multiply*J_vals*(-1)
            new_total_energy = np.sum(before_summed)
            #print(new_normalised_coords)
            #deciding whether the energy is accepted or not 
            if new_total_energy <= total_energy:
                energy_vals.append(total_energy)
                energy_vals_cont.append(total_energy)
                accepted_energy_change.append(abs(new_total_energy - total_energy))
                total_energy = new_total_energy 
                #print(np.sum(new_normalised_coords**2))
                normalised_spins = new_normalised_spins
                #print(np.sum(normalised_coords**2))
                acceptance += 1
                count += 1
                d0_when_energy_change.append(d0)
            else: 
                rejection += 1
                energy_vals_cont.append(total_energy)
                count += 1
                #print(np.sum(new_normalised_coords**2))
                #ADD IN A DEFLATION SCHEDULE 
                
            if count == thousand_monte_carlo_steps:
                d0 = d0-1 
                if d0 == 0:
                    #print('breaking')
                    break
                #print(str(d0) + ' is now our starting value for d0')
                count = 0 
                monte_carlo_step = round(N/d0)
                steps = 100
                thousand_monte_carlo_steps = monte_carlo_step*steps
                #print(thousand_monte_carlo_steps)
    graph_energy.append(total_energy)
    print('Sample Finished')
            
    return energy_vals, acceptance, rejection, acceptance, energy_vals_cont, total_energy, graph_energy, d0_starting_vals, d0_when_energy_change, accepted_energy_change

def mean_graph_energy(d0_starting_vals, mean_energy, var_energy):
    plt.scatter(d0_starting_vals, mean_energy)
    plt.errorbar(d0_starting_vals, mean_energy, yerr= var_energy, linestyle="None")
    plt.title('P=3 Spin Glass Model \n Average Final Energies per Particle vs Initial Move-Class Value \n The Number of Samples Taken for Averages is %i, Number of MC Steps is 100 \n Linear Deflation Schedule' %samples)
    plt.ylabel('Final Average Energy per Spin')
    plt.xlabel('d0 Start Value')
    return plt.show()

def final_energy_per_spin(d0_vals, final_energy_per_spin, final_energy_per_spin_var):
    plt.scatter(d0_vals, final_energy_per_spin)
    plt.errorbar(d0_vals, final_energy_per_spin, yerr=final_energy_per_spin_var, linestyle="None")
    plt.title('P=3 Spin Glass Model, Final Average Energy per Spin')
    plt.ylabel('Final Average Energy per Spin')
    plt.xlabel('d0')
    return plt.show()

def energy_change(acceptance):
    acceptances = []
    for i in range(acceptance + 1):
        acceptances.append(i)
    plt.plot(acceptances, energy_vals)
    plt.title('P=3 Spin Glass Model')
    plt.ylabel('Energy Change')
    plt.xlabel('Acceptance Number')
    return plt.show()
    
def energy_change_cont(thousand_monte_carlo_steps):
    continuous = []
    for i in range(thousand_monte_carlo_steps+1):
        continuous.append(i)
    plt.plot(continuous, energy_vals_cont)
    plt.title('P=3 Spin Glass Model')
    plt.ylabel('Energy Change')
    plt.xlabel('Move Number')
    return plt.show()

#just flattens the list f lists we get above 
def flatten(t):
    return [item for sublist in t for item in sublist]

def avg_accepted_moves_per_step_hist(keys, vals):
    plt.bar(keys, (np.divide(list(vals), samples)), width = 0.3)
    plt.ylabel('Average Number of Accepted Moves')
    plt.xlabel('Step Size')
    plt.title('Average Number of Accepted Moves per Step Size\n Averaged over %i Samples \n Linear Deflation Schedule' %samples)
    return plt.show()

def avg_energy_per_accepted_move_hist(bins, avg_val):
    plt.bar(bins, avg_val, width = 0.3)
    plt.ylabel('Average Energy Change')
    plt.xlabel('d0')
    plt.title('Average Energy Change per Accepted Move\n Averaged over %i Samples \n Linear Deflation Schedule' %samples)
    return plt.show()

input_list = [set_up(d0) for i in range(samples)]

#this is going to be the multiprocessing code that will speed up our simulations 
energy_array = []
final_energy = []

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cores) as pool:
        final_results = pool.starmap(OMCD, input_list)
        energy_vals_cont = [x[4] for x in final_results]
        energy_vals = [x[0] for x in final_results]
        acceptance = [x[1] for x in final_results]
        graph_energy = [x[6] for x in final_results]
        d0_starting_vals = [x[7] for x in final_results]
        d0_when_energy_change = [x[8] for x in final_results]
        accepted_energy_change = [x[9] for x in final_results]
        d0_starting_vals = d0_starting_vals[0]
        graph_energy = np.asarray(graph_energy)
        graph_energy = graph_energy/N
        mean_energy = np.mean(graph_energy, axis=0)
        var_energy = np.var(graph_energy, axis=0)
        
        flattened_d0_when_energy_change = flatten(d0_when_energy_change)
        flattened_accepted_energy = flatten(accepted_energy_change)
        accepted_moves_dict = Counter(flattened_d0_when_energy_change)
        keys = accepted_moves_dict.keys()
        vals = accepted_moves_dict.values()
        zipped = zip(flattened_d0_when_energy_change, flattened_accepted_energy)
        d = {}
        for tuple in zipped:
            key,val = tuple
            d.setdefault(key, []).append(val)
        
        bins = []
        avg_val = []
        for hist_val, values in d.items():
            bins.append(hist_val)
            avg = sum(values)/len(values)
            avg_val.append(avg)
            
        avg_energy_per_accepted_move_hist(bins, avg_val)
        avg_accepted_moves_per_step_hist(keys, vals)


end = time.time()
print('This program took ' + str(end-start) + ' to run.')


#%%

bins = []
variances = []
means = []
for hist_val, values in d.items():
    bins.append(hist_val)
    var = np.var(values)
    mean = np.mean(values)
    variances.append(var)
    means.append(mean)
    
print(means)
print(variances)

#%%
d1_count = []
for i in d0_when_energy_change:
    counter = 0
    for j in i:
        if j==1:
            counter += 1
    d1_count.append(counter)

d1_count_cummulative = np.cumsum(d1_count)
d1_count_cummulative = d1_count_cummulative.insert(0,0)
#%%
lowest_bin_vals = []
for hist_val, values in d.items():
    if hist_val == 1:
        lowest_bin_vals.append(values)

lowest_bin_vals = flatten(lowest_bin_vals)
nums = []
for i in range(len(lowest_bin_vals)):
    nums.append(i)
    
plt.scatter(nums, lowest_bin_vals, s=3)
plt.ylabel('Energy Change')
plt.xlabel('Index of Accepted Move at d0 = 1')
plt.title('Examining the Energy Change Over \n Time at the Lowest Move-Class Value\n Averaged over %i Samples \n Linear Deflation Schedule' %samples)
for xc in d1_count_cummulative:
    plt.axvline(x=xc, color='r', linewidth = 0.5, label = 'New Sample')
plt.show()