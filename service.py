from helper import *


class Service:

    def __init__(self, serviceID):
        self.serviceID = serviceID  # int
        self.vulnerability = Helper.assign_prob_of_exploit()  # float
        self.used = False

    def __str__(self):
        return "Service ID = " + str(self.serviceID) + \
               " Vulnerability score = " + str(round(self.vulnerability, 3))

    @staticmethod
    def initialise_service(no_services):
        all_services = []
        for i in range(no_services):
            all_services.append(Service(i))
        return all_services

