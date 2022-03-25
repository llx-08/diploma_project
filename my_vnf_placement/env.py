from os import link
from pdb import main
from select import select
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# 参数设置：large/small
# 主机CPU个数
CPU_PROPERTIES_SMALL       = [10, 9, 8, 7, 6, 6, 6, 6, 6, 6]
CPU_PROPERTIES_LARGE       = [10, 9, 8, 7, 6, 6, 6, 6, 6, 6, 10, 9, 8, 7, 6, 6, 6, 6, 6, 6]

# 链路带宽
LINK_PROPERTIES_BW_SMALL   = [1000, 1000, 500, 400, 100, 100, 100, 100, 100, 100]
LINK_PROPERTIES_BW_LARGE   = [1000, 1000, 500, 400, 100, 100, 100, 100, 100, 100, 1000, 1000, 500, 400, 100, 100, 100,
                            100, 100, 100]

# 链路延迟
LINK_PROPERTIES_LAT_SMALL  = [30, 50, 10, 50, 50, 50, 50, 50, 50, 50]
LINK_PROPERTIES_LAT_LARGE  = [30, 50, 10, 50, 50, 50, 50, 50, 50, 50, 30, 50, 10, 50, 50, 50, 50, 50, 50, 50]

# VNFD大小/带宽/延迟
VNFD_PROPERTIES_SIZE_SMALL = [0, 4, 3, 3, 2, 2, 2, 1, 1]
VNFD_PROPERTIES_BW_SMALL   = [0, 100, 80, 60, 20, 20, 20, 20, 20]
VNFD_PROPERTIES_LAT_SMALL  = [0, 100, 80, 60, 20, 20, 20, 20, 20]


class Environment(object):


    def __init__(self, num_cpus, num_vnfds, env_profile="small_default", dict_vnf_profile="small_default"):

        # 环境内参数初始化
        self.num_cpus         = num_cpus
        self.num_vnfds        = num_vnfds
        self.cpu_properties   = [{"numSlots": 0} for _ in range(num_cpus)]  # numSlots：可以理解为每个主机拥有的CPU个数？
        self.link_properties  = [{"bandwidth": 0, "latency": 0} for _ in range(num_cpus)]  # 链路属性初始化
        self.vnfds_properties = [{"size": 0, "bandwidth": 0, "latency": 0} for _ in range(num_vnfds+1)]  # vnfd属性初始化

        # 暂时不知道 ：？
        self.p_min  = 200
        self.p_slot = 100

        # 环境内参数赋值/VNFD参数赋值
        self._getEnvProperties(num_cpus, env_profile)
        self._getVnfdProperties(num_vnfds, dict_vnf_profile)

        # cell 存储放置的vnf信息（编号）
        self.max_slots = max([cpu["numSlots"] for cpu in self.cpu_properties])
        self.cells = np.empty((self.num_cpus, self.max_slots))


        self._initEnv()


    def _initEnv(self):

        # 初始化使用的cpu/link为0
        self.cells[:]  = np.nan
        self.cpu_used  = np.zeros(self.num_cpus)
        self.link_used = np.zeros(self.num_cpus)

        # 将部署相关信息初始化
        self.service_length = 0
        self.network_service = None
        self.placement = None
        self.first_slots = None
        self.reward = None
        
        # 约束变量
        self.constraint_occupancy = None
        self.constraint_bandwidth = None
        self.constraint_latency = None
        self.invalid_placement = False
        self.invalid_bandwidth = False
        self.invalid_latency = False

        self.link_latency = 0
        self.cpu_latency = 0


    def _getEnvProperties(self, num_cpus, env_profile):
        '''根据env_profile=large/small 初始化环境内相关变量'''
        if env_profile == "small_default":
            
            for i in range(num_cpus):
                self.cpu_properties[i]["numSlots"]   = CPU_PROPERTIES_SMALL[i]
                self.link_properties[i]["bandwidth"] = LINK_PROPERTIES_BW_SMALL[i]
                self.link_properties[i]["latency"]   = LINK_PROPERTIES_LAT_SMALL[i]
        
        elif env_profile == "large_default":

            for i in range(num_cpus):
                self.cpu_properties[i]["numSlots"]   = CPU_PROPERTIES_LARGE[i]
                self.link_properties[i]["bandwidth"] = LINK_PROPERTIES_BW_LARGE[i]
                self.link_properties[i]["latency"]   = LINK_PROPERTIES_LAT_LARGE[i]

        else:
            raise Exception('Environment Not Detected, Please Choose The Right Property')

            
    def _getVnfdProperties(self, num_vnfds, dict_vnf_profile):
        '''dict_vnf_profile=large/small 初始化vnf内相关变量'''
        if dict_vnf_profile == "small_default":

            for i in range(num_vnfds+1):  # 为什么+1？
                self.vnfds_properties[i]["size"]      = VNFD_PROPERTIES_SIZE_SMALL
                self.vnfds_properties[i]["bandwidth"] = VNFD_PROPERTIES_BW_SMALL
                self.vnfds_properties[i]["latency"]   = VNFD_PROPERTIES_LAT_SMALL

        else:
            raise Exception("VNF dict not detected, plz choose the right property")

    
    def _placeSlot(self, cpu, vnf):
        '''在当前CPU内寻找可以放置vnf中slot的主机(core?)'''
        occupied_slot = np.nan

        for slot in range(self.cpu_properties[cpu]["numSlots"]):
            if np.isnan(self.cells[cpu][slot]):
                self.cells[cpu][slot] = vnf
                occupied_slot = slot
                break

        return occupied_slot

    def _placeVNF(self, i, cpu, vnf):
        '''放置VNF'''

        # 如果vnf大小小于当前CPU剩余的槽位，即能够放置于当前CPU
        if self.vnfds_properties[vnf]["size"] < (self.cpu_properties[cpu]["numSlots"] - self.cpu_used[cpu]):
            for slot in range(self.vnfds_properties[vnf]["size"]):
                occupied_slot = self._placeSlot(cpu, vnf)

                # 如果是第一个Slot，进行注释（标记？）
                if slot == 0:
                    self.first_slots[i] = occupied_slot

            # 如果当前cpu内槽位够用的话肯定会把vnf内都分配，因此cpu_used直接加上vnf_size
            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]  

        else:  # 无法承载对应vnf数量的slots

            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]
            self.first_slots[i] = -1

    def _computeLink(self):
        '''计算链路使用率和链路延时'''

        # 计算链路带宽只需要考虑最大的
        self.bandwidth = max([self.vnfd_properties[vnf]["bandwidth"] for vnf in self.network_service])

        for i in range(self.service_length):
            cpu = self.placement[i]

            if i == 0:  # service chain 中第一个vnf？还是vnf中第一个slot
                self.link_used[cpu] += self.bandwidth
                self.link_latency   += self.link_properties[cpu]["latency"]
            elif cpu != self.placement[i-1]:  
                # 如果当前cpu和上一个vnf使用的不是一个cpu，那么就需要加上它们通讯所需的链路带宽
                self.link_used[cpu] += self.bandwidth
                self.link_latency   += self.link_properties[cpu]["latency"]

            if i == self.service_length-1:  # service chain 中最后一个
                self.link_used[cpu] += self.bandwidth
                self.link_latency   += self.link_properties[cpu]["latency"]
            elif cpu != self.placement[i+1]:
                # 与上文同理
                self.link_used[cpu] += self.bandwidth
                self.link_latency   += self.link_properties[cpu]["latency"]

    def _computeReward(self):

if __name__ == "__main__":
    pass