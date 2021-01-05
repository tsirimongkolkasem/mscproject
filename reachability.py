#NOT BEING USE ATM
import numpy as np
import matplotlib.pyplot as plt
class Reachability:
    def __init__(self, rules, default_policy):
        self.rules = rules
        self.default_policy = default_policy


    def __str__(self):
        None


class Rule:
    def __init__(self, source, destination, ports,action):
        self.source = source
        self.destination = destination
        self.ports = ports
        self.action = action


if __name__ =='__main__':
    k = [1,2,3,4,5,6,7]
    k_legend = ["k=1","k=2","k=3","k=4","k=5","k=6","k=7"]
    M = 1000
    N = [0,1000,1500,2000,3000,5000,6000,7878,10000]
    y = [0,3.9,9.5,    13,30.9,76.4,110.25,237,380]
    # for i in k:
    #     y.append((np.exp((N * i) / M) - 1) / (np.exp(i) - 1))
    #
    # for i in range(len(k)):
    #     plt.plot(N,y[i])
    plt.plot(N,y)
    plt.xlabel('Number of Enclaves')
    plt.ylabel('Time/s')
    plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
    #plt.legend(k_legend)
    plt.show()
    pass