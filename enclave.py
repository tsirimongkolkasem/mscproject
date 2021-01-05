import device
import numpy as np
class Enclave:
    def __init__(self, enclaveID, vulnerability,devices=[], tap_sensitivity = 0.75):
        self.enclaveID = enclaveID                  # int
        self.tap_sensitivity = tap_sensitivity      # Float
        self.vulnerability = vulnerability          # Float
        self.devices = devices                      # []
        self.uncomp_devices = devices               # []
        self.comp_devices = []                      # []
        self.n = len(devices)                       # int
        self.no_compromised = 0                     # int
        self.mission_delay = 0                      # int
        self.resource_loss = 0                      # float


##### Methods used in Erik Hemberg's paper in
    # modelling devices infection within an enclave
    def spread_malware(self,beta,t):

        if(np.random.random() < self.vulnerability):
            total_comp_devices = np.floor(self.n*np.exp(beta*t)/(self.n + (np.exp(beta*t)-1)))
            new_comp_devices = total_comp_devices - self.no_compromised

            randomlist = list(range(len(self.uncomp_devices)))
            np.random.shuffle(randomlist)
            for i in range(new_comp_devices):
                self.uncomp_devices[randomlist[i]].is_compromised = True
                if self.uncomp_devices[randomlist[i]].is_resource_device:
                    self.resource_loss = self.resource_loss + 5
                self.comp_devices.append(self.uncomp_devices[randomlist[i]])

            self.no_compromised = total_comp_devices
            self.uncomp_devices = [device for device in self.uncomp_devices if not device.is_compromised]


    def detect_compromise(self):
        threshold = self.no_compromised * self.tap_sensitivity

        return np.random.uniform(0,self.n) < threshold


    def cleanse_enclave(self):
        for d in self.devices:
            if d.mission_device:            #if d is a mission device
                self.mission_delay += 1     #update mission delay
            d.compromised = False           #cleanse device

    def reset_enclave(self):
        np.random.shuffle(self.devices)
        self.uncomp_devices = self.devices
        self.comp_devices = np.zeros(self.n)

    def update_mission_delay(self):
        for d in self.devices:
            if d.mission_device:            #if d is a mission device
                self.mission_delay += 1     #update mission delay



