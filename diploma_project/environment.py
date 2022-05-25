from os import link
from pdb import main
from select import select
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# 参数设置：large/small
# 主机CPU个数
CPU_PROPERTIES_SMALL = [10, 9, 8, 7, 6, 6, 6, 6, 6, 6]
CPU_PROPERTIES_LARGE = [10, 9, 8, 7, 6, 6, 6, 6, 6, 6, 10, 9, 8, 7, 6, 6, 6, 6, 6, 6]

# 链路带宽
LINK_PROPERTIES_BW_SMALL = [1000, 1000, 500, 400, 100, 100, 100, 100, 100, 100]
LINK_PROPERTIES_BW_LARGE = [1000, 1000, 500, 400, 100, 100, 100, 100, 100, 100, 1000, 1000, 500, 400, 100, 100, 100,
                            100, 100, 100]

# 链路延迟
LINK_PROPERTIES_LAT_SMALL = [30, 50, 10, 50, 50, 50, 50, 50, 50, 50]
LINK_PROPERTIES_LAT_LARGE = [30, 50, 10, 50, 50, 50, 50, 50, 50, 50, 30, 50, 10, 50, 50, 50, 50, 50, 50, 50]

# VNFD大小/带宽/延迟
VNFD_PROPERTIES_SIZE_SMALL = [0, 4, 3, 3, 2, 2, 2, 1, 1]
VNFD_PROPERTIES_BW_SMALL = [0, 100, 80, 60, 20, 20, 20, 20, 20]
VNFD_PROPERTIES_LAT_SMALL = [0, 100, 80, 60, 20, 20, 20, 20, 20]


class Environment(object):

    def __init__(self, num_cpus, num_vnfds, env_profile="small_default", dict_vnf_profile="small_default"):

        # 环境内参数初始化
        self.num_cpus = num_cpus
        self.num_vnfds = num_vnfds
        self.cpu_properties = [{"numSlots": 0} for _ in range(num_cpus)]  # numSlots：可以理解为每个主机拥有的CPU个数？
        self.link_properties = [{"bandwidth": 0, "latency": 0} for _ in range(num_cpus)]  # 链路属性初始化
        self.vnfd_properties = [{"size": 0, "bandwidth": 0, "latency": 0} for _ in range(num_vnfds + 1)]  # vnfd属性初始化

        # 计算reward/cost 服务器运行成本
        self.p_min = 200
        self.p_slot = 100

        # 环境内参数赋值/VNFD参数赋值
        self._getEnvProperties(num_cpus, env_profile)
        self._getVnfdProperties(num_vnfds, dict_vnf_profile)

        # cell 存储放置的vnf信息（编号）
        self.max_slots = max([cpu["numSlots"] for cpu in self.cpu_properties])
        self.cells = np.empty((self.num_cpus, self.max_slots))

        # Network Function相关
        self.service_length = None
        self.network_service = None
        self.placement = None
        self.first_slots = None

        self._initEnv()

    def _initEnv(self):

        # 初始化使用的cpu/link为0
        self.cells[:] = np.nan
        self.cpu_used = np.zeros(self.num_cpus)
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
        """根据env_profile=large/small 初始化环境内相关变量"""
        if env_profile == "small_default":

            for i in range(num_cpus):
                self.cpu_properties[i]["numSlots"] = CPU_PROPERTIES_SMALL[i]
                self.link_properties[i]["bandwidth"] = LINK_PROPERTIES_BW_SMALL[i]
                self.link_properties[i]["latency"] = LINK_PROPERTIES_LAT_SMALL[i]

        elif env_profile == "large_default":

            for i in range(num_cpus):
                self.cpu_properties[i]["numSlots"] = CPU_PROPERTIES_LARGE[i]
                self.link_properties[i]["bandwidth"] = LINK_PROPERTIES_BW_LARGE[i]
                self.link_properties[i]["latency"] = LINK_PROPERTIES_LAT_LARGE[i]

        else:
            raise Exception('Environment Not Detected, Please Choose The Right Property')

    def _getVnfdProperties(self, num_vnfds, dict_vnf_profile):
        """dict_vnf_profile=large/small 初始化vnf内相关变量"""
        if dict_vnf_profile == "small_default":

            for i in range(num_vnfds + 1):  # +1
                self.vnfd_properties[i]["size"] = VNFD_PROPERTIES_SIZE_SMALL[i]
                self.vnfd_properties[i]["bandwidth"] = VNFD_PROPERTIES_BW_SMALL[i]
                self.vnfd_properties[i]["latency"] = VNFD_PROPERTIES_LAT_SMALL[i]

        else:
            raise Exception("VNF dict not detected, plz choose the right property")

    def _placeSlot(self, cpu, vnf):
        """在当前CPU内寻找可以放置vnf中slot的主机"""
        occupied_slot = np.nan

        for slot in range(self.cpu_properties[cpu]["numSlots"]):
            if np.isnan(self.cells[cpu][slot]):
                self.cells[cpu][slot] = vnf
                occupied_slot = slot
                break

        return occupied_slot

    def _placeVNF(self, vnf_i, cpu, vnf):
        """放置VNF"""

        # print("place VNF:")
        # print(vnf)
        # print(self.vnfd_properties[vnf])
        # print(self.vnfd_properties[vnf]["size"])
        # print(self.cpu_properties[cpu]["numSlots"])
        # print(self.cpu_used[cpu])

        # 如果vnf大小小于当前CPU剩余的槽位，即能够放置于当前CPU
        if self.vnfd_properties[vnf]["size"] <= (self.cpu_properties[cpu]["numSlots"] - self.cpu_used[cpu]):

            for slot in range(self.vnfd_properties[vnf]["size"]):
                occupied_slot = self._placeSlot(cpu, vnf)

                # 如果是第一个Slot，进行标记
                if slot == 0:
                    self.first_slots[vnf_i] = occupied_slot

            # 如果当前cpu内槽位够用的话肯定会把vnf内都分配，因此cpu_used直接加上vnf_size
            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]

        else:  # 无法承载对应vnf数量的slots
            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]
            self.first_slots[vnf_i] = -1

    def _computeLink(self):
        """计算链路使用率和链路延时"""

        # 计算链路带宽按照最大的进行计算
        self.bandwidth = max([self.vnfd_properties[vnf]["bandwidth"] for vnf in self.network_service])

        for i in range(self.service_length):
            cpu = self.placement[i]
            # 由于我们的拓扑结构是星形拓扑，因此每两个不在同一台服务器上的VNF都需要有一条从第一个VNF出来的链路和进入第二个VNF的链路
            # 因此我们要分两种情况进行考虑，同时计算链路成本
            # 考虑入VNF
            if i == 0:  # service chain 中第一个vnf，不需要考虑链路相关
                pass
                # self.link_used[cpu] += self.bandwidth
                # self.link_latency += self.link_properties[cpu]["latency"]
            elif cpu != self.placement[i - 1]:
                # 如果当前cpu和上一个vnf使用的不是一个cpu，那么就需要加上它们通讯所需的链路带宽
                self.link_used[cpu] += self.bandwidth
                self.link_latency += self.link_properties[cpu]["latency"]

            # 考虑出VNF
            if i == self.service_length - 1:  # service chain 中最后一个
                pass
                # self.link_used[cpu] += self.bandwidth
                # self.link_latency += self.link_properties[cpu]["latency"]
            elif cpu != self.placement[i + 1]:
                # 与上文同理
                self.link_used[cpu] += self.bandwidth
                self.link_latency += self.link_properties[cpu]["latency"]

    def _computeReward(self):
        """计算奖励 & 检查各项约束条件是否满足"""

        # 检查占用
        self.constraint_occupancy = 0
        for i in range(self.num_cpus):
            if self.cpu_used[i] > self.cpu_properties[i]["numSlots"]:  # 如果使用的CPU大于CPU可用的slot数量
                self.invalid_placement = True
                self.constraint_occupancy += self.cpu_used[i] - self.cpu_properties[i]["numSlots"]

        # 检查带宽
        self.constraint_bandwidth = 0
        for i in range(self.num_cpus):
            if self.link_used[i] > self.link_properties[i]["bandwidth"]:
                self.invalid_bandwidth = True
                self.constraint_bandwidth += self.link_used[i] - self.link_properties[i]["bandwidth"]

        # 检查延迟
        self.cpu_latency = sum(
            [self.vnfd_properties[vnf]["latency"] for vnf in self.network_service[:self.service_length]])

        self.constraint_latency = 0
        if self.link_latency > self.cpu_latency:
            self.invalid_latency = True
            self.constraint_latency += self.link_latency - self.cpu_latency

        # 计算Reward
        self.reward = 0
        for cpu in range(self.num_cpus):
            if self.cpu_used[cpu]:
                self.reward += self.p_min + self.p_slot * self.cpu_used[cpu]

    def step(self, length, network_service, placement):
        """放置网络功能(Network Service)"""
        self.service_length = length
        self.network_service = network_service
        self.placement = placement
        self.first_slots = -np.ones(length, dtype='int32')  # 初始化为-1，代表没有first slot

        for i in range(length):  # 放置功能链中每个VNF
            self._placeVNF(i, placement[i], network_service[i])

        self._computeLink()
        self._computeReward()

    def clear(self):
        """重新初始化"""
        self._initEnv()

    def is_err(self):
        """检查是否是正确的放置策略"""
        for f in self.first_slots:
            if f == -1:
                return False
        return True

    def render(self):
        """ MatplotLib 绘图 """

        fig, ax = plt.subplots()
        ax.set_title('Environment')

        margin = 3
        margin_ext = 6
        xlim = 100
        ylim = 80

        plt.xlim(0, xlim)
        plt.ylim(-ylim, 0)

        high = np.floor((ylim - 2 * margin_ext - margin * (self.num_cpus - 1)) / self.num_cpus)
        wide = np.floor((xlim - 2 * margin_ext - margin * (self.max_slots - 1)) / self.max_slots)

        plt.text(1, 0, "Energy: {}".format(self.reward), ha="center", family='sans-serif', size=8)
        plt.text(1, 2, "Cstr occ: {}".format(self.constraint_occupancy), ha="center", family='sans-serif', size=8)
        plt.text(20, 0, "Cstr bw: {}".format(self.constraint_bandwidth), ha="center", family='sans-serif', size=8)
        plt.text(20, 2, "Cstr lat: {}".format(self.constraint_latency), ha="center", family='sans-serif', size=8)

        # slot序号
        for slot in range(self.max_slots):
            x = wide * slot + slot * margin + margin_ext
            plt.text(x + 0.5 * wide, -3, "slot{}".format(slot), ha="center", family='sans-serif', size=8)

        # cpu 序号
        for cpu in range(self.num_cpus):
            y = -high * (cpu + 1) - (cpu) * margin - margin_ext
            plt.text(0, y + 0.5 * high, "cpu{}".format(cpu), ha="center", family='sans-serif', size=8)

            for slot in range(self.cpu_properties[cpu]["numSlots"]):
                x = wide * slot + slot * margin + margin_ext
                rectangle = mp.Rectangle((x, y), wide, high, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rectangle)  # 添加矩形补丁（slot框）

        cmap = plt.cm.get_cmap('gist_rainbow')
        colormap = [cmap(np.float32(i + 1) / (self.service_length + 1)) for i in range(self.service_length)]

        # 画放置的VNF
        for idx in range(self.service_length):
            vnf = self.network_service[idx]
            cpu = self.placement[idx]
            first_slot = self.first_slots[idx]

            for k in range(self.vnfd_properties[vnf]["size"]):

                if first_slot != -1:  # 每个VNF的第一个如果是-1是有问题的
                    slot = first_slot + k
                    x = wide * slot + slot * margin + margin_ext
                    y = -high * (cpu + 1) - cpu * margin - margin_ext
                    rectangle = mp.Rectangle((x, y), wide, high, linewidth=0, facecolor=colormap[idx], alpha=.9)
                    ax.add_patch(rectangle)
                    plt.text(x + 0.5 * wide, y + 0.5 * high, "vnf{}".format(vnf), ha="center", family='sans-serif',
                             size=8)

        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    num_cpus = 10
    num_vnfds = 8

    env = Environment(num_cpus, num_vnfds)

    # test
    service_length = 8
    network_service = [4, 8, 1, 4, 3, 6, 6, 8]
    placement = [3, 3, 2, 1, 1, 0, 0, 0]

    env.step(service_length, network_service, placement)
    print("is err:", env.is_err())
    print("first slot")
    print(env.first_slots)
    print("Placement Invalid: ", env.invalid_placement)
    print("Link used: ", env.link_used, "Invalid: ", env.invalid_bandwidth)
    print("CPU Latency: ", env.cpu_latency, "Link Latency: ", env.link_latency, "Invalid: ", env.invalid_latency)
    print("Energy: ", env.reward)
    print("Constraint_occupancy: ", env.constraint_occupancy)
    print("Constraint_bandwidth: ", env.constraint_bandwidth)
    print("Constraint_latency: ", env.constraint_latency)

    env.render()
    env.clear()
