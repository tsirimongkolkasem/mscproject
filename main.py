from helper import *
from network_model import Network_model as nm
from enclave import *
from reachability import Reachability
from device import *
from service import Service
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import copy
from particle import *
import gc


def main(dataset):
    gc.collect()
    tap_sensitivity = 0

    no_services = 10

    no_devices = 50
    no_resource_devices = 2
    no_mission_devices = 10
    start_time = time.time()
    max_devices_per_enclave = 50


    # initialise services
    all_services = Service.initialise_service(no_services)

    # TODO grouping devices and traffic requirements
    # traffic requirements, node u, node v, service s
    # Note: node 0 is the internet
    # e.g. [(u1,v1,s1),(u2,v2,s2),...]
    # e.g. (0,1,3) means enclave 1 has to be connected to enclave 0 (the internet) via service 3
    """
    Case:
    0 = internet 
    1 = Web server via FW1 (= Service 0, CVE????)
    
    
    Service
    
    """
    service_score = []
    for i in range(len(all_services)):
        service_score.append(all_services[i].vulnerability)

    all_services[0].vulnerability = 0.950
    all_services[1].vulnerability = 0.950
    all_services[2].vulnerability = 0.502
    all_services[3].vulnerability = 0.898
    all_services[4].vulnerability = 0.113
    all_services[5].vulnerability = 0.250
    all_services[6].vulnerability = 0.362
    all_services[7].vulnerability = 0.945
    all_services[8].vulnerability = 0.442
    all_services[9].vulnerability = 0.671


    traffic_requirements = [(0,1,0),(2,3,1)]#,(2,3,8),(4,5,7),(5,2,6),(2,8,1)]#,(5,0,7)]#(4,0,3),(3,0,3)]
    # Compute minimum number of enclaves required, given the traffic requirements

    min_enclaves = nm.compute_min_enclave(traffic_requirements)

    max_iter = 0





    all_devices = []
    """
    for i in range(number_of_particles):
         particle_devices = \
            nm.initialise_devices(no_devices, no_mission_devices, no_resource_devices)
         all_devices.append(particle_devices)
    """

    global_best_score = -np.inf
    global_best_position = None

    number_of_particles, update_iter,max_enclaves = 20, 10, 100

    # alpha sec_score + beta cost_score
    alpha, beta, k = 0.5, 0.5, 7

    # initialise Particle Swarm Optimisation parameters
    learning_rate, W, phi_p, phi_g = 0.5, 0.7, 1.5, 1.5

    #learning_rate, W, phi_p, phi_g = 0.5, 1, 1.5, 1.5


    # Legend for plots
    legend = [k,alpha,beta,learning_rate,W, phi_p, phi_g,number_of_particles]

    population = []
    for i in range(number_of_particles):

        graph = nm.initialise_model(no_devices, max_enclaves,min_enclaves,
                                    traffic_requirements,all_services,
                                    max_iter)
        network_mod = nm(graph,max_enclaves,traffic_requirements,
                         min_enclaves,all_services,k=k)
        particle = Particle(network_mod, learning_rate)
        fitness_score = particle.compute_fitness(alpha,beta)
        if fitness_score > global_best_score:
            global_best_score = fitness_score
            global_best_position = copy.deepcopy(particle.current_position)

        population.append(particle)

    # Plot fitness for particle initialisation
    Particle.plot_fitness(population, max_enclaves,legend,alpha,beta)

    # Initialise global best position
    Particle.update_swarm_best_position(population,global_best_position)

    #W, phi_p, phi_g = 0.2, 1, 0.5
    average_fitness_per_iteration = []
    print('best = ' + str(len(global_best_position.best_graph)))
    for i in range(update_iter):
        for particle in population:
            # for Xi in particle.velocity: Note: note need atm as
            # current velocity only have 1 dimension, which is the number of nodes

            # Pick r_p,r_g ~ U(0,1)
            r_p , r_g = np.random.uniform(0,1), np.random.uniform(0,1)
            particle.update_velocity(W,phi_p,phi_g,r_p,r_g,global_best_position)

            # Update particle's position
            particle.update_position()

            # Compute particle's fitness
            particle.compute_fitness(alpha,beta)
            # compute fitness also update personal best score
            # ^(if fitness > personal best)

            # if personal best score > global best score
            if particle.fitness_score > global_best_score:
                global_best_score = particle.fitness_score
                global_best_position = copy.deepcopy(particle.current_position)
                Particle.update_swarm_best_position(population
                                                    ,global_best_position)

        average = Particle.plot_fitness(population,max_enclaves,legend,alpha,beta)
        average_fitness_per_iteration.append((average))


    Helper.plot_list(average_fitness_per_iteration)

    #Helper.plot_graph(particle.network_model.best_graph)
    end_time = time.time()
    print("Computation time = "+str(end_time-start_time)+" seconds")
    print("Best score = "+str(global_best_score))
    print('best = ' + str(len(global_best_position.best_graph)-1))
    Helper.plot_graph(global_best_position.best_graph)

    Helper.plot_list_services(all_services)

    return 0


if __name__ == "__main__":
    import sys

    main(sys.argv[0])
