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

        # 环境内参数设置
        self.num_cpus       = num_cpus
        self.num_vnfds      = num_vnfds
        self.cpu_properties = [{"numSlots": 0} for _ in range(num_cpus)]  # numSlots：可以理解为每个主机拥有的CPU个数？
