from network_model import *
from enclave import *
from device import *
from helper import *
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Particle:

    def __init__(self, network_model, lr):
        self.lr = lr
        self.current_position = network_model
        self.fitness_score = -np.inf

        self.velocity = self.initialise_velocity()

        self.global_best_position = None

        self.personal_best_score = network_model.personal_best_score
        self.personal_best_position = copy.deepcopy(network_model)


        self.global_best_score = -np.inf

    def compute_fitness(self,alpha,beta):
        self.fitness_score = self.current_position.compute_fitness(alpha,beta)
        if self.fitness_score > self.personal_best_score:
            self.personal_best_score = self.fitness_score
            self.personal_best_position = copy.deepcopy(self.current_position)

            if self.personal_best_score > self.global_best_score:
                self.global_best_position = self.personal_best_position

        return self.fitness_score


    def update_velocity(self,W,phi_p,phi_g,r_p,r_g,g_d):

        self.velocity = W *self.velocity+phi_p*r_p*(len(self.personal_best_position.best_graph)\
                        -len(self.current_position.graph))+phi_g*r_g*(len(g_d.best_graph)-len(self.current_position.graph))
        self.velocity = np.floor(self.velocity)

    def update_position(self):

        changes = self.current_position.change_architecture(self.velocity)
        self.velocity = changes

    def initialise_velocity(self):
        min_e = self.current_position.min_enclaves
        max_e = self.current_position.max_enclaves
        node_velocity = np.random.randint(-(max_e-min_e), max_e-min_e)
        return node_velocity

    @staticmethod
    def plot_fitness(population,xmax,legend,alpha,beta):

        k,alpha,beta,lr,W,phi_p,phi_g,n = \
            legend[0],legend[1],legend[2],legend[3],legend[4],legend[5] \
            ,legend[6],legend[7]

        fitness_score =[]
        enclave_no = []
        for particle in population:
            fitness_score.append(particle.compute_fitness(alpha,beta))
            enclave_no.append(len(particle.current_position.graph)-1)

        fig, ax = plt.subplots(1,figsize=(5,5))

        legend_elements = [ Line2D([0], [0], color='b', marker=None, label="k = "+str(k))\
                           ,Line2D([0], [0], color='b', marker=None, label="$w_{sec}$ = "+str(alpha))\
                           ,Line2D([0], [0], color='b', marker=None, label="$w_{cost}$ = "+str(beta))\
                           ,Line2D([0], [0], color='b', marker=None, label="lr = "+str(lr))\
                           ,Line2D([0], [0], color='b', marker=None, label="$W$ = "+str(W))\
                           ,Line2D([0], [0], color='b', marker=None, label="$\phi_p$ = "+str(phi_p))\
                           ,Line2D([0], [0], color='b', marker=None, label="$\phi_g$ = "+str(phi_g))\
                           ,Line2D([0], [0], color='b', marker=None, label="$n$ = "+str(n))
                            ]
        ax.legend(handles=legend_elements, loc='lower right',handlelength=0)

        ax.set_xlabel('Number of Enclaves')
        ax.set_xlim((0,xmax))
        ax.set_ylabel('Fitness Score')
        ax.set_ylim((-1,0))
        ax.plot(enclave_no, fitness_score, 'b.')
        plt.show()

        mean = np.mean(fitness_score)
        return mean

    @staticmethod
    def update_swarm_best_position(population,global_best_position):
        for particle in population:
            particle.global_best_position = copy.deepcopy(global_best_position)
