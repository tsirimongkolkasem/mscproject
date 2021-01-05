
class Device:
    def __init__(self, deviceID, is_compromised=False, is_mission_device=False, is_resource_device=False):
        self.deviceID = deviceID
#        self.ports = ports
        self.is_compromised = is_compromised
        self.is_mission_device = is_mission_device
        self.is_resource_device = is_resource_device
        self.connected_to_dID = None

    def __str__(self):
        return ""


