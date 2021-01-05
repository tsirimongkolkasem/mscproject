import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import string

# from service import Service

class Helper:
    def __init__(self):
        pass

    @staticmethod
    def assign_prob_of_exploit():
        """
        Assign a random CVSS score from CVSS distribtuion
        Return a probability that an enclave is compromised
        (Conditionally depends on the enclave before being compromised)
        """

        population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        prob = [0.0060, 0.0070, 0.0400, 0.0370, 0.2220,
                0.1930, 0.1380, 0.2220, 0.0040, 0.1310]
        score = random.choices(population, prob)[0]
        if score == 10:
            score -= 0.05

        # uniform distribution U~(score-1,score)
        score = np.random.uniform(score - 1, score)
        return score / 10

    @staticmethod
    def plot_graph(graph):
        n = len(graph)
        min_graph_size = 10
        graph_size = min_graph_size
        if n > min_graph_size:
            graph_size = 10 + (n-10)*0.5
        fig, ax = plt.subplots(2,figsize=(graph_size,graph_size))
        ax1 = ax[0]

        pos = nx.planar_layout(graph)

        for i in range(len(graph)):
            # if i == 0:
            #     p = 1
            # else:
            if i == 0:
                #p = 1
                pass
            else:
                pass
            p = graph.nodes[i]['P']

            p = round(p,3)
            x,y = pos[i]
            ax1.text(x+0.01,y+0.01,s=str(p))

        # ax1.figure(1,figsize=(10,10))
        nx.draw_networkx_nodes(graph, pos,ax=ax1, node_size=300)  # node_size=700
        nx.draw_networkx_edges(graph, pos, width=1.5,ax=ax1)
        nx.draw_networkx_labels(graph, pos, font_size=n//10+13, font_family="sans-serif",ax=ax1)
        # edge_labels = nx.get_edge_attributes(graph,'weight')

        edge_labels = dict([((u, v,), f"{d['weight']:.3f}") for u, v, d in graph.edges(data=True)])

        nx.draw_networkx_edge_labels(graph, pos, label_pos=0.5, edge_labels=edge_labels, font_size=n//10+12,
                                     font_family="sans-serif",ax=ax1)
        # nx.draw(graph,with_labels=True)

        ax2 = ax[1]

        # ax2 = fig.add_subplot(2,2,1)
        # ax2.figure(2,figsize=(10,10))
        nx.draw_networkx_nodes(graph, pos,ax=ax2)  # node_size=700
        nx.draw_networkx_edges(graph, pos, width=1.5,ax=ax2)
        nx.draw_networkx_labels(graph, pos)  # font_size=10, font_family="sans-serif")
        # edge_labels = nx.get_edge_attributes(graph,'weight')
        edge_labels = dict([((u, v,), f"{d['label']}") for u, v, d in graph.edges(data=True)])
        nx.draw_networkx_edge_labels(graph, pos, label_pos=0.3, edge_labels=edge_labels, font_size=n//10+15,
                                     font_family="sans-serif",ax=ax2)
        # nx.draw(graph,with_labels=True)


        fig,ax = plt.subplots(1,figsize=(graph_size/2,graph_size/2))

        nx.draw_networkx_nodes(graph, pos, node_size=220)  # node_size=700
        nx.draw_networkx_edges(graph, pos, width=1.5)
        nx.draw_networkx_labels(graph, pos)  # font_size=10, font_family="sans-serif")
        edge_labels = dict([((u, v,), f"{d['label']}") for u, v, d in graph.edges(data=True)])
        nx.draw_networkx_edge_labels(graph, pos, label_pos=0.5, edge_labels=edge_labels, font_size=n // 10 + 16,
                                     font_family="sans-serif", ax=ax)
        plt.savefig("Optimum_Architecture.jpg")
        plt.show()

    @staticmethod
    def plot_list_services(all_services):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('off')

        style = dict(size=10, color='gray')
        y = len(all_services)
        i = 0
        for service in all_services:
            if i % 10 == 0:
                y = len(all_services)
            text = str(service)
            ax.text(i//10*20, y, text)
            y -= 1
            i += 1
        ax.axis([1,  len(all_services), 1,  len(all_services)])
        plt.show()

    @staticmethod
    def plot_list(list_):
        ymin = min(list_)
        plt.figure()
        plt.xlabel('Number of Iterations')
        plt.ylim(ymin,0)
        #plt.xticks(np.arange(0,len(list_)+1,1))
        plt.ylabel('Average Fitness Score')
        plt.plot(list_, 'b.')
        plt.show()

    @staticmethod
    def convert_to_ascii(num):
        num2alphadict = dict(zip(range(1, 27), string.ascii_lowercase))
        outval = ""
        numloops = (num - 1) // 26

        if numloops > 0:
            outval = outval + Helper.convert_to_ascii(numloops)
        remainder = num % 26
        if remainder > 0:
            outval = outval + num2alphadict[remainder]
        else:
            outval = outval + "z"
        return outval

    @staticmethod
    def convert_to_directed(graph,digraph,from_u=0):
#        visited[from_u] = True
        G = nx.DiGraph()
        visited, queue = set(),[from_u]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                for u,v,d in graph.edges(vertex,data=True):
                    if visited[v]:
                        pass
                    else:
                        p = graph.edges[u,v]['weight']
                        labell = graph.edges[u,v]['label']
                        G.add_edge(u, v, weight=p, label=labell)

                    queue.append(v)

        return G
"""
        def BFS(graph, start):
            visited, queue = set(), [start]
            while queue:
                vertex = queue.pop(0)
                if vertex not in visited:
                    visited.add(vertex)
                    for u, v in graph.edges(vertex):
                        queue.append(v)
            return list(visited)
"""
