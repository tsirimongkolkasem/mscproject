from enclave import *
from device import Device
from helper import *
import copy
from belief_propagation.belief_propagation import *


class Network_model:


    def __init__(self, G, max_enclaves,traffic_requirements,min_enclaves,all_services,k):
        self.fitness_score_history = []     # Record fitness score for the model
        self.all_services = all_services
        self.min_enclaves = min_enclaves
        self.max_enclaves = max_enclaves
        self.traffic_requirements = traffic_requirements
        self.graph = G                      # Current connections
        self.best_graph = copy.deepcopy(G)  # Best configuration
        self.E = self.initialise_enclave()  # All enclaves
#        self.m = 0                         # Mission delay
#        self.beta = 1
        self.personal_best_score = -np.inf
        self.k = k                          # Steepness constant
#       self.reachability = reachability

    def __str__(self): #print the whole network out
        return "" #print out fitness?

    def initialise_enclave(self):
        enclave = []
        for i in range(len(self.graph)):
            prob_enclave_comp = self.graph.nodes[i]['P']
            enclave.append(Enclave(i,prob_enclave_comp))
        return enclave

    def change_architecture(self, d_n):
        """
        :param
        d_n: Change in number of nodes
        :return:
        Network architecture with the change in number of nodes
        and also different configuration of services and enclaves
        whilst following the traffic requirements
        """
        current_number_of_enclaves = len(self.graph)-1
        no_nodes = current_number_of_enclaves + d_n
        message = 0         #return the difference

        if no_nodes < self.min_enclaves:
            d_n = self.min_enclaves -current_number_of_enclaves
            message = d_n
            no_nodes = self.min_enclaves
        elif no_nodes > self.max_enclaves:
            d_n = self.max_enclaves-no_nodes # compute difference between
                                             # max number and current enclaves no
            message = d_n
            no_nodes = self.max_enclaves
        else:
            message = d_n
        no_nodes = int(no_nodes)
        new_graph = Network_model.initialise_model(100,self.max_enclaves,self.min_enclaves,
                                       self.traffic_requirements,self.all_services,2,
                                       random =False,n=no_nodes)
        self.graph = new_graph
        Network_model.compute_marginal_two(self.graph)
        self.E = self.initialise_enclave()
        return message

    def compute_fitness(self,alpha,beta):
        security_score = self.compute_security_score()
        cost_score = self.compute_cost_score()
        fitness_score = alpha*security_score+beta*cost_score
        if (fitness_score> self.personal_best_score):
            self.personal_best_score = fitness_score
            self.best_E = copy.deepcopy(self.E)
            self.best_graph = nx.graph.deepcopy(self.graph)
        self.fitness_score_history.append(fitness_score)
        return fitness_score

    def compute_security_score(self):
        security_score = 0
        for e in self.E:
            if e.enclaveID == 0:
                pass
            else:
                security_score += e.vulnerability

        security_score = security_score/(len(self.E)-1)
        return -security_score

    def compute_cost_score(self):
        N = len(self.graph)-1
        M = self.max_enclaves
        k = self.k
        return -(np.exp((N*k)/M)-1)/(np.exp(k)-1)

    # Methods used in Erik Hemberg's paper in
    # modelling devices infection within an enclave
    # note: function not used in the project

    def run_simulation(self, maxT):
        total_mission_delay = 0
        for t in range(1, maxT):
            for enclave in self.E:
                enclave.spread_malware(self.beta, t)
                need_cleansing = enclave.detect_compromise()
                if need_cleansing:
                    enclave.mission_delay += enclave.cleanse_enclave()
                    enclave.reset_enclave()
                else:
                    enclave.mission_delay += enclave.update_mission_delay()

    @staticmethod
    def DFS(adj_mat, start, visited):
        visited[start] = True
        for i in range(len(visited)):
            if (adj_mat[start][i] != 0 and (not visited[i])):
                Network_model.DFS(adj_mat,i,visited)
        return

    @staticmethod
    def BFS(graph, start):
        visited, queue = set(), [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                for u,v in graph.edges(vertex):
                    queue.append(v)
        return list(visited)

    @staticmethod
    def add_edge(G, u, v, s, all_services):
        try:
            # if weight is already assigned
            G.edges[u,v]['weight'] =\
                1-(1-G.edges[u,v]['weight'])*(1-all_services[s].vulnerability)
            G.edges[u,v]['label'].append(all_services[s].serviceID)
        except:
            G.add_edge(
                u, v, weight=all_services[s].vulnerability,
                label=[all_services[s].serviceID])
        all_services[s].used = True
        return

    @staticmethod
    def initialise_edges(no_enclaves, traffic_requirements, all_services):

        G = nx.Graph()
        for nodeID in range(no_enclaves+1):
            G. add_node(nodeID)

        #Helper.plot_graph(G)
        for u,v,s in traffic_requirements:
            Network_model.add_edge(G,u,v,s,all_services)
            #Helper.plot_graph(G)

        connected = [False] * (no_enclaves + 1)
        connected_nodes = Network_model.BFS(G, 0)
        list__ = list(range(len(connected)))
        np.random.shuffle(list__)
        # connect unconnected node to a connected one randomly
        for i in list__:
            if i not in connected_nodes:
                #add_edge
                unused_service = Network_model.services_unused(all_services)
                r = np.random.randint(0,len(connected_nodes))
                if not unused_service:
                    Network_model.add_edge(G,i,connected_nodes[r],np.random.randint(0,len(all_services)),all_services)
                else:
                    np.random.  shuffle(unused_service)
                    Network_model.add_edge(G,i,connected_nodes[r],unused_service[0],all_services)

                connected_nodes = Network_model.BFS(G, 0)
                #Helper.plot_graph(G)

        return G


    @staticmethod
    def services_unused(all_services):
        """
        :param all_services:

        :return: list of services that have not been used yet

        This function has defense in depth principle in mind,
        avoiding single point of failure
        """
        temp = []
        for i in range(len(all_services)):
            if not all_services[i].used:
                temp.append(i)
        return temp

    @staticmethod
    def initialise_model(no_devices, max_no_enclaves, min_enclaves,traffic_requirements, all_services, max_iter,
                         devices_per_enclave=1000, no_resource_devices=2, no_mission_devices=10,random=True,n=0):
        for service in all_services:
            service.used = False
        #if min_enclaves > 9:
        # randomly generate number of enclaves
        if random:
            no_enclaves = np.random.randint(min_enclaves, max_no_enclaves)
        else:
            no_enclaves = n
        """
        if (no_enclaves * devices_per_enclave < no_devices):
            raise ValueError("Not enough enclaves for all devices")
        # initialise devices (for E. Hemberg and N. Wagner paper)
        
        all_devices = \
            Network_model.initialise_devices(no_devices, no_mission_devices, no_resource_devices)
        
        # Initialise enclave devices (not enclave object)
        # (may not need for enclave level)
        # may be used in the future for computing traffic requirements when devices are grouped into enclaves
        # and certain devices have certain requirements and properties
        
        part_devices = \
            Network_model.initialise_enclave_devices(all_devices, no_devices, no_enclaves, devices_per_enclave)
        
        # for i in range(no_enclaves):
        #    tap_sensitivity = np.random.uniform(0, 1.5)

        #    Enclave(i,tap_sensitivity,part_devices)
        #    pass
        """
        # initialise connections (edges)
        # add core edges first (traffic requirement)
        graph = Network_model.initialise_edges(no_enclaves, traffic_requirements, all_services)

        # Create factor graph and compute marginal probabilities for each enclave
        Network_model.compute_marginal_two(graph)

        return graph

    @staticmethod
    def compute_min_enclave(traffic_requirements):
        """
        :param traffic_requirements:
        :return: minimum number of enclaves require
                 for the network model
        """
        if not traffic_requirements:
            return 1
        required_enclaves = []
        for u, v, s in traffic_requirements:
            required_enclaves.append(u)
            required_enclaves.append(v)
        required_enclaves = set(required_enclaves)
        min_enclaves = len(required_enclaves)
        if max(required_enclaves) >= min_enclaves:
            min_enclaves = max(required_enclaves)
        if 0 in set(required_enclaves):
            min_enclaves -= 1
        return min_enclaves

    @staticmethod
    def compute_marginal_two(graph):
        """
        :param a fully connected graph with weights for all edges,
               weights for this experiment represent the conditional
               probability of compromising an enclave
        :return:a fully connected graph with marginal probability
                of compromising each enclave
        """
        fg = Network_model.create_factor_graph_two(graph)

        try:
            nx.algorithms.find_cycle(graph)
            lbp = loopy_belief_propagation(fg)
            for i in range(len(graph)):
                p = lbp.belief('n'+str(i),10).get_distribution()[1]
                graph.nodes[i]['P'] = p
        except:
            bp = belief_propagation(fg)
            for i in range(len(graph)):
                p = bp.belief('n'+str(i)).get_distribution()[1]
                graph.nodes[i]['P'] = p


        return

    @staticmethod
    def create_factor_graph_two(graph):

        cycles = []
        try:
            cycle = nx.algorithms.find_cycle(graph)
            cycles = set([item for sublist in cycle for item in sublist])
        except:
            pass

        pgm_1 = factor_graph()
        #f1 = factor(['n' + str(0)], np.array([0, 1]))
        #pgm_1.add_factor_node('0', f1)
        queue = []
        n_node = len(graph)
        visited = [False] * n_node
        initialised = [False] *n_node
        queue.append(0)
        visited[0] = True
        initialised[0] = True
        # use DFS to create factor graph
        # while queue is not empty
        while queue:
            i = queue.pop()
            visited[i] = True
            for u, v in graph.edges(i):
                facname_u = Helper.convert_to_ascii(u)
                facname_v = Helper.convert_to_ascii(v)
                p = graph.edges[u, v]['weight']
                if not visited[v]:

                    if u == 0:

                        # node 0 is the Internet, thus it will not be
                        # affected by other nodes since it is assumed to be
                        # compromised.
                        # 0 was not used since it will throw division by 0 error

                        pgm_1.add_factor_node('f'+facname_v + facname_u,
                                              factor(['n'+str(v),'n'+str(u)],np.array([[1, 1-p],[0,p]])))
                        #ff = factor(['n' + str(v),'n' + str(u)], np.array([[0, 1-p],
                        #                                                    [0, p]]))

                        initialised[v] = p

                        """
                        pgm_1.add_factor_node('f' + facname_u + facname_v,
                                              factor(['n' + str(v), 'n' + str(u)], np.array([
                                                  [1-p, p],
                                                  [1-p, p]])))
                                                  """
                        #pgm_1.add_factor_node('f'+facname_v, factor([str('n') + str(v)], np.array([1-p, p])))
                        #pgm_1.change_factor_distribution('f'+facname_v,f1)


                    else:
                        if (not initialised[v]) and (u not in cycles) and (v not in cycles):
                            pgm_1.add_factor_node('f' + facname_v + facname_u,
                                                  factor(['n' + str(v), 'n' + str(u)], np.array([[1, 1 - p],[0, p]])))
                            initialised[v] = p
                        else:
                            initialised[v] = 1 - ((1 - p) * (1 - initialised[v]))

                        # If xv is not compromised, then the node next to it
                        # will not be compromised, again 0 was not used due to division by 0 error
                        #if (not initialised[v])  and (v not in cycles):
                           # pgm_1.add_factor_node('f'+facname_v + facname_u,
                           #                       factor(['n'+str(v),'n'+ str(u)], np.array([[1, 1-p],[0, p]]) ))
                           # initialised[v] = p
                        #else:
                            #pass
                            """
                            else:
                                pgm_1.add_factor_node('f' + facname_v + facname_u,
                                                      factor(['n' + str(v), 'n' + str(u)], np.array([
                                                          [1, 1 - (1 - ((1 - p) * (1 - initialised[v])))],
                                                          [0, 1 - ((1 - p) * (1 - initialised[v]))]])))
                                initialised[v] = p
                            """
                            """
                            pgm_1.add_factor_node('f' + facname_v + facname_u,
                                                  factor(['n' + str(v), 'n' + str(u)], np.array([
                                                      [initialised[v], 1 - (1 - ((1 - p) * (1 - initialised[v])))],
                                                      [1 - initialised[v], 1 - ((1 - p) * (1 - initialised[v]))]])))

                            """


                    queue.append(v)
        plot_factor_graph(pgm_1)
        return pgm_1

    @staticmethod
    def initialise_devices(no_devices,no_mission_devices,no_resource_devices):
        all_devices = []
        for i in range(no_devices):
            all_devices.append(Device(i))
        for i in range(no_mission_devices):
            all_devices[i].is_mission_device = True
        for i in range(no_resource_devices):
            all_devices[i].is_resource_device = True
        np.random.shuffle(all_devices)
        return all_devices

    @staticmethod
    def initialise_enclave_devices(all_devices,no_devices,no_enclaves,devices_per_enclave):
        if (no_enclaves * devices_per_enclave < no_devices):
            raise ValueError()

        n_devices = len(all_devices)
        all_np_devices = np.asarray(all_devices, dtype=object)
        list_of_devices_in_enclave = np.random.dirichlet(np.ones(no_enclaves), size=1)[0]
        list_of_devices_in_enclave = np.rint(np.multiply(list_of_devices_in_enclave, n_devices))
        list_of_devices_in_enclave = [int(i) for i in list_of_devices_in_enclave]
        # Ensure no enclaves have 0 device (needs improvement)

        for i in range(len(list_of_devices_in_enclave)):
            if list_of_devices_in_enclave[i] == 0:
                list_of_devices_in_enclave[i] += 1
                max_device_index = \
                    list_of_devices_in_enclave.index(max(list_of_devices_in_enclave))
                list_of_devices_in_enclave[max_device_index] -= 1


        #print(list_of_devices_in_enclave)

        np.random.shuffle(all_np_devices)
        i = 0
        part_devices = []

        for num in list_of_devices_in_enclave:
            num = int(num)
            part_devices.append(all_np_devices[i:i+num])
            i = i+num
        return part_devices

