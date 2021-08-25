# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 18:32:45 2021

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


#size of the lattice 
#NOTE: P spin does not have to be a lattice so we need to change this code so it is not an array and just a list. 
nx = 3 
p=3

#number of particles 
N = nx**3
d0 = 12

def set_up(d0):
    #generating a random list of coords with a gaussian dist N[0, 1]
    coords = [[[np.random.normal(loc = 0.0, scale = 1) for i in range(nx)] for j in range(nx)] for k in range(nx)]
    coords = np.asarray(coords)
    
    #we need to normalize the coords so that the squared sum of them = N so this is what this does. 
    N_inverse = 1/N
    squared_coords = coords**2
    sum_square_coords = np.sum(squared_coords)
    factor = sum_square_coords*N_inverse
    inverse_factor = 1/math.sqrt(factor)
    
    #Note: due to floating point arithmetic in python, the sum of the normalised coords may be slghtly more or less than N 
    original_normalised_coords = coords*inverse_factor 
    
    #just flattening the array to make the next bit easier 
    flattened = original_normalised_coords.flatten()
    
    #now need to create a probability dist of J values to associate with each triplet 
    #this is the gaussian dist that J is taken from 
    variance = math.factorial(p)/((2*N)^(p-1))
    strd_dev = math.sqrt(variance)
    J_dist = np.random.normal(loc = 0.0, scale = strd_dev)
    
    #listing the possible triplets for the p-spin model 
    triplets = list(itertools.combinations(flattened, 3))
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
    normalised_coords = original_normalised_coords.copy()
    return thousand_monte_carlo_steps, normalised_coords, total_energy, acceptance, rejection, J_vals, N_inverse, strd_dev, d0

samples = 200

def OMCD(thousand_monte_carlo_steps, normalised_coords, total_energy, acceptance, rejection, J_vals, N_inverse, strd_dev, d0):
    d0_starting_vals = [x for x in range(1, d0+1)]
    d0_starting_vals.reverse()
    graph_energy = []
    first_energy = total_energy 
    first_coords = normalised_coords
    first_J_vals = J_vals 
    first_N_inverse = N_inverse
    for ix in d0_starting_vals:
        print('Starting value of d0 is now ' + str(ix))
        d0 = ix
        #total_energy = first_energy     
        energy_vals = [total_energy]
        energy_vals_cont = [total_energy]
        count = 0
        total_energy = first_energy
        normalised_coords = first_coords 
        J_vals = first_J_vals
        N_inverse = first_N_inverse
        
        while d0 > 0: 
        #for i in range(100):
            
            new_normalised_coords = normalised_coords.copy()
            
            #randomly choosing d0 amount of coords to change the spin of
            change_coords = []
            for x in range(0, nx):
                for y in range(0, nx):
                    for z in range(0, nx):
                        change_coords.append([x, y, z])
            
            random.shuffle(change_coords)
          
            #this is the sample of coords we will change 
            change_these = change_coords[0:d0]
            
            #this part will change the spins of the particles chosen 
            changing = []
            
            for j in change_these:
                a = j[0]
                b = j[1]
                c = j[2]
                changing.append(new_normalised_coords[a][b][c])
             
            #perturbing the selected particles 
            for i in range(len(changing)):
                changing[i] = changing[i] + np.random.normal(loc = 0.0, scale = strd_dev)
                
            changing = np.asarray(changing)
        
            
            #need to insert these new perturbed changes back into the original array 
            for i in range(len(change_these)): 
                a = change_these[i][0]
                b = change_these[i][1]
                c = change_these[i][2]
                new_normalised_coords[a][b][c] = changing[i]
        # =============================================================================
        #     print(changing)
        #     print(normalised_coords)
        # =============================================================================
            
            #we have to renormalize them in this part  
            squared_change = new_normalised_coords**2
            sum_square_change = np.sum(squared_change)
            change_factor = sum_square_change*N_inverse
            inverse_change_factor = 1/math.sqrt(change_factor)   
            new_normalised_coords = new_normalised_coords*inverse_change_factor
            
            
            #now we need to recalulate the energy, accept/reject it and do the process over and over again 
            #doing the same procedure as before 
            new_flattened = new_normalised_coords.flatten()
            new_triplets = list(itertools.combinations(new_flattened, 3))
            spins_multiply = [(x*y*z) for x, y, z in new_triplets]
            spins_multiply = np.asarray(spins_multiply) 
            before_summed = spins_multiply*J_vals*(-1)
            new_total_energy = np.sum(before_summed)
            #print(new_normalised_coords)
            #deciding whether the energy is accepted or not 
            if new_total_energy <= total_energy:
                energy_vals.append(total_energy)
                energy_vals_cont.append(total_energy)
                total_energy = new_total_energy 
                #print(np.sum(new_normalised_coords**2))
                normalised_coords = new_normalised_coords
                #print(np.sum(normalised_coords**2))
                acceptance += 1
                count += 1
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
                print(str(d0) + ' is now our starting value for d0')
                count = 0 
                monte_carlo_step = round(N/d0)
                steps = 100
                thousand_monte_carlo_steps = monte_carlo_step*steps
                #print(thousand_monte_carlo_steps)
        graph_energy.append(total_energy)
            
    return energy_vals, acceptance, rejection, acceptance, energy_vals_cont, total_energy, graph_energy, d0_starting_vals

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

# =============================================================================
# #this is to try and repeat the process over many samples to find final average energy per particle 
# final_energy_per_spin = []
# final_energy_per_spin_var = []
# d0_vals = []
# #need to change this to a multiprocessing loop so it does the samples quicker 
# energy_array = []
# final_energy = []
# d0_vals.append(d0)
# for i in range(samples):
#     set_up_vals = set_up(d0)
#     final_results = OMCD(set_up_vals[0], set_up_vals[1], set_up_vals[2], set_up_vals[3], set_up_vals[4], set_up_vals[5], set_up_vals[6], set_up_vals[7], set_up_vals[8])
#     energy_vals_cont = final_results[4]
#     energy_vals = final_results[0]
#     acceptance = final_results[1]
#     graph_energy = final_results[6]
#     energy_array.append(graph_energy)
#     thousand_monte_carlo_steps = set_up_vals[0]
#     final_energy.append(final_results[5])
#     d0_starting_vals = final_results[7]
#     print('Sample ' + str(i+1)+ '/' + str(samples) + ' Complete')
# final_energy = [x / N for x in final_energy]
# energy_array = np.asarray(energy_array)
# energy_array = energy_array/N
# mean_energy = np.mean(energy_array, axis=0)
# var_energy = np.var(energy_array, axis=0)
# =============================================================================

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
        d0_starting_vals = d0_starting_vals[0]
        graph_energy = np.asarray(graph_energy)
        graph_energy = graph_energy/N
        mean_energy = np.mean(graph_energy, axis=0)
        var_energy = np.var(graph_energy, axis=0)
        mean_graph_energy(d0_starting_vals, mean_energy, var_energy)
        

end = time.time()
print('This program took ' + str(end-start) + ' to run.')