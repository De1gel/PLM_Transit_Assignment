import numpy as np
import itertools
import copy
import matplotlib.pyplot as plt


# 改动：原算法比较cinv，ALS取某一种策略，策略的所有子策略在MSA中均
# 增加流量。ef去掉了小于零的情况
class Network():
    def __init__(self):
        self.epsilon = 0.0000001
        self.zones = ('A', 'X', 'Y', 'B')
        self.lines = ('L1', 'L2', 'L3', 'L4')
        self.scheduled_frequency = {line: 10 for line in self.lines}
        self.vehicle_capacity = {line: 100 for line in self.lines}
        self.vehicle_capacity['L1'] = 120
        self.vehicle_capacity['L4'] = 80
        self.scheduled_frequency['L3'] = 15
        self.scheduled_frequency['L4'] = 20
        self.itinerary = {
            'L1': [['A', 'B'], [25]],
            'L2': [['A', 'X', 'Y'], [7, 6]],
            'L3': [['X', 'Y', 'B'], [4, 4]],
            'L4': [['Y', 'B'], [10]]
        }
        self.itinerary_matrix = self.init_itinerary_matrix()
        # 初始化OD需求
        self.demand = {
            'A': {'B': 1600},
            'X': {'B': 1300}
        }

    def init_itinerary_matrix(self):
        matrix = np.zeros((len(self.zones), len(self.zones)))
        itinerary_matrix = [matrix.copy() for i in range(len(self.lines))]
        for line in self.lines:
            itinerary = self.itinerary[line]
            for i in range(len(itinerary[0]) - 1):
                origin = self.zones.index(itinerary[0][i])
                destination = self.zones.index(itinerary[0][i + 1])
                itinerary_matrix[self.lines.index(line)][origin][destination] = 1
        return itinerary_matrix

    def get_FLS_FNS(self, origin, destination, max_transfer_time):
        FLS = []
        FNS = {}
        for transfer_time in range(max_transfer_time, -1, -1):
            for zones in list(itertools.combinations(self.zones, transfer_time)):
                if origin in zones or destination in zones:
                    continue
                # 生成换乘方案
                node_list = [origin]
                for i in range(transfer_time):
                    node_list.append(zones[i])
                node_list.append(destination)
                line_combinations = list(itertools.combinations(self.lines, transfer_time + 1))
                for lines in line_combinations:
                    # if node_list[1] == 'X' and lines[0] == "L2" and origin == 'X' and destination == 'B' and max_transfer_time == 2:
                    #     print(node_list)
                    feasibility = 1
                    for i in range(len(lines)):
                        sum_row = sum(self.itinerary_matrix[self.lines.index(lines[i])][self.zones.index(node_list[i])])
                        sum_column = sum(self.itinerary_matrix[self.lines.index(lines[i])][:, self.zones.index(node_list[i + 1])])
                        feasibility = feasibility * (sum_row * sum_column)
                    if feasibility:
                        if lines[0] not in FLS:
                            FLS.append(lines[0])
                            FNS[lines[0]] = []
                        if node_list[1] not in FNS[lines[0]]:
                            # if node_list[1] == 'X' and lines[0] == "L2" and origin == 'X' and destination == 'B' and max_transfer_time == 2:
                            #     print(node_list)
                            FNS[lines[0]].append(node_list[1])
        return (FLS, FNS)

class LNS():
    def __init__(self, transfer_time):
        self.transfer_time = transfer_time
        self.FNS_FLS_all = None
        self.decision_porpotion = None
        self.decision_flow = None
        self.efreq_line = None
        self.ANS_ALS_all = None
        self.FNS = None
        self.FLS = None
        self.c_ttt_set = None
        self.c_inv_set = None
        self.min_c_ttt = None
        self.min_c_inv = None
        self.v = None

    def init_decision_porpotion(self, net):
        Lambda_line = {}
        Lambda_node = {}
        for origin in net.zones:
            for destination in net.zones:
                for t in range(self.transfer_time, -1, -1):
                    if origin != destination:
                        lst = self.FNS_FLS_all[(origin, destination, t)]
                        FLS_subsets = [subset for i in range(1, len(lst[0]) + 1) for subset in itertools.combinations(lst[0], i)]
                        for FLS_subset in FLS_subsets:
                            Lambda_line[(origin, destination, t, FLS_subset)] = 1 / len(FLS_subsets)
                        for line in lst[0]:
                            for node in lst[1][line]:
                                Lambda_node[(origin, destination, t, line, node)] = 1 / len(lst[1][line])
        self.decision_porpotion = (Lambda_line, Lambda_node)

    def calculate_flow(self, net):
        def init_flow(origin, destination, transfer_time):
            if transfer_time == self.transfer_time:
                try:
                    flow = net.demand[origin][destination]
                except Exception as e:
                    flow = 0
            else:
                flow = sum(flow_ for key, flow_ in flow_node.items() if key[1] == destination and key[4] == origin and key[2] == transfer_time + 1)
            return flow
        flow_lineset = {}
        flow_line = {}
        flow_node = {}
        flow = {}
        for t in range(self.transfer_time, -1, -1):
            for origin in net.zones:
                for destination in net.zones:
                    if origin != destination:
                        flow[(origin, destination, t)] = init_flow(origin, destination, t)
                        lst = self.FNS_FLS_all[(origin, destination, t)]
                        FLS_subsets = [subset for i in range(1, len(lst[0]) + 1) for subset in itertools.combinations(lst[0], i)]
                        for FLS_subset in FLS_subsets:
                            flow_lineset[(origin, destination, t, FLS_subset)] = flow[(origin, destination, t)] * self.decision_porpotion[0][(origin, destination, t, FLS_subset)]
                        for line in lst[0]:
                            flow_line_propotion = sum(self.efreq_line[(line, origin)] * self.decision_porpotion[0][(origin, destination, t, lineset)] / sum(self.efreq_line[(line_prime, origin)] for line_prime in lineset) for lineset in FLS_subsets if line in lineset)
                            flow_line[(origin, destination, t, line)] = flow[(origin, destination, t)] * flow_line_propotion
                            for node in lst[1][line]:
                                flow_node[(origin, destination, t, line, node)] = flow_line[(origin, destination, t, line)] * self.decision_porpotion[1][(origin, destination, t, line, node)]

        self.decision_flow = (flow_lineset, flow_node, flow_line)

    def init_FNS_FLS_all(self, net):
        FNS_FLS_all = {}
        for origin in net.zones:
            for destination in net.zones:
                if origin != destination:
                    for t in range(self.transfer_time + 1):
                        FNS_FLS_all[(origin, destination, t)] = net.get_FLS_FNS(origin, destination, t)
        self.FNS_FLS_all = FNS_FLS_all
        self.FNS = {key: value[1] for key, value in FNS_FLS_all.items()}
        self.FLS = {key: value[0] for key, value in FNS_FLS_all.items()}
        self.efreq_line = {}
        for line in net.lines:
            for node in net.itinerary[line][0]:
                self.efreq_line[(line, node)] = net.scheduled_frequency[line]

    def efreq_calculation(self, net):
        def eff_function(v, line):
            Numerator = net.scheduled_frequency[line] * 1
            saturation_rate = v / (net.scheduled_frequency[line] * net.vehicle_capacity[line])
            Denumenator = 1 + 3 * saturation_rate ** 11
            return Numerator / Denumenator
        v_a = {}
        v_b = {}
        v_0 = {}
        efreq_line = {}
        for line in net.lines:
            for node in net.itinerary[line][0]:
                v_a[(line, node)] = 0
                v_b[(line, node)] = 0
                v_0[(line, node)] = 0
                for (origin, destination, t, line_, node_) in self.decision_flow[1].keys():
                    if line_ == line and node_ == node:
                        v_a[(line, node)] += self.decision_flow[1][(origin, destination, t, line_, node_)]
                for (node_, destination, t, line_) in self.decision_flow[2].keys():
                    if line_ == line and node_ == node:
                        v_b[(line, node)] += self.decision_flow[2][(node_, destination, t, line_)]
            for node in net.itinerary[line][0]:
                for node_ in net.itinerary[line][0]:
                    if net.itinerary[line][0].index(node_) < net.itinerary[line][0].index(node):
                        v_0[(line, node)] += v_b[(line, node_)] - v_a[(line, node_)]
                # if v_b[(line, node)] > net.scheduled_frequency[line] * net.vehicle_capacity[line] - v_0[(line, node)] - net.epsilon:
                #     efreq_line[(line, node)] = net.scheduled_frequency[line] * net.epsilon / (v_b[(line, node)] + net.epsilon)
                # else:
                #     efreq_line[(line, node)] = net.scheduled_frequency[line] * (net.scheduled_frequency[line] * net.vehicle_capacity[line] - v_0[(line, node)] - v_b[(line, node)]) / (net.scheduled_frequency[line] * net.vehicle_capacity[line] - v_0[(line, node)])
                # if v_0[(line, node)] + v_b[(line, node)] - v_a[(line, node)] > net.scheduled_frequency[line] * net.vehicle_capacity[line]:
                #     expected_waiting_time = (v_0[(line, node)] + v_b[(line, node)] - v_a[(line, node)] - net.scheduled_frequency[line] * net.vehicle_capacity[line]) * (net.scheduled_frequency[line] + 1)
                #     efreq_line[(line, node)] = 1 / (expected_waiting_time + 1 / net.scheduled_frequency[line])
                # else:
                #     efreq_line[(line, node)] = net.scheduled_frequency[line]
                efreq_line[(line, node)] = eff_function(v_0[(line, node)] + v_b[(line, node)] - v_a[(line, node)], line)
        self.efreq_line = efreq_line
        self.v = [v_a, v_b, v_0]

    def ANS_ALS_calculation(self, net):
        def c_inv_calculation(origin, destination, t, line):
            min_c_inv[(origin, destination, t, line)] = float('inf')
            for node in self.FNS[(origin, destination, t)][line]:
                inv_time = sum(net.itinerary[line][1][net.itinerary[line][0].index(origin):net.itinerary[line][0].index(node)])
                c_inv = 0.5 + inv_time + c_ttt_calculation(node, destination, t - 1, self.FLS)  # 此处考虑了dwell time
                c_inv_set[(origin, destination, t, line, node)] = c_inv
                if c_inv < min_c_inv[(origin, destination, t, line)]:
                    min_c_inv[(origin, destination, t, line)] = c_inv
                    min_node = node
            ANS[(origin, destination, t)][line] = min_node
            return min_c_inv[(origin, destination, t, line)]

        def c_ttt_calculation(origin, destination, t, ALS):
            min_c_ttt[(origin, destination, t)] = float('inf')
            if origin == destination:
                min_c_ttt[(origin, destination, t)] = 0
                return 0
            line_subsets = [subset for i in range(1, len(ALS[(origin, destination, t)]) + 1) for subset in itertools.combinations(ALS[(origin, destination, t)], i)]
            for lineset in line_subsets:
                # 计算有效发车频率
                efreq = sum(self.efreq_line[(line, origin)] for line in lineset) / 60
                # 计算在车时间
                in_vehicle_time = sum(self.efreq_line[(line, origin)] / 60 * c_inv_calculation(origin, destination, t, line) for line in lineset)
                # 计算期望行程时间
                c_ttt = (1 + in_vehicle_time) / efreq
                c_ttt_set[(origin, destination, t, lineset)] = c_ttt
                if c_ttt < min_c_ttt[(origin, destination, t)]:
                    min_c_ttt[(origin, destination, t)] = c_ttt
            return min_c_ttt[(origin, destination, t)]
        c_inv_set = {}
        c_ttt_set = {}
        min_c_inv = {}
        min_c_ttt = {}
        ALS = copy.deepcopy(self.FLS)
        ANS = copy.deepcopy(self.FNS)
        for origin in net.zones:
            for destination in net.zones:
                for t in range(self.transfer_time, -1, -1):
                    if origin != destination:
                        c_ttt_calculation(origin, destination, t, self.FLS)
        self.c_ttt_set = c_ttt_set
        self.c_inv_set = c_inv_set
        self.min_c_ttt = min_c_ttt
        self.min_c_inv = min_c_inv
        for origin in net.zones:
            for destination in net.zones:
                for t in range(self.transfer_time, -1, -1):
                    if origin != destination:
                        # 按c_TTT对线路进行排序
                        lines_sorted = sorted(self.FLS[(origin, destination, t)], key=lambda x: c_ttt_set[(origin, destination, t, (x,))])
                        # 生成吸引线路集
                        ALS[(origin, destination, t)] = []
                        if len(lines_sorted) > 0:
                            ALS[(origin, destination, t)].append(lines_sorted[0])
                        c_ttt_calculation(origin, destination, t, ALS)
                        for k in range(1, len(lines_sorted)):
                            AL = {}
                            AL[((origin, destination, t))] = [lines_sorted[0]]
                            if c_ttt_calculation(origin, destination, t, AL) < min_c_ttt[(origin, destination, t)]:
                                ALS[(origin, destination, t)].append(lines_sorted[k])
                            else:
                                break
                            c_ttt_calculation(origin, destination, t, ALS)
        self.ANS_ALS_all = (ANS, ALS)

    def decision_porpotion_update(self, net, aux_propotion, i):
        for key in self.decision_porpotion[0].keys():
            self.decision_porpotion[0][key] = self.decision_porpotion[0][key] + (aux_propotion[0][key] - self.decision_porpotion[0][key]) / (i + 1)
        for key in self.decision_porpotion[1].keys():
            self.decision_porpotion[1][key] = self.decision_porpotion[1][key] + (aux_propotion[1][key] - self.decision_porpotion[1][key]) / (i + 1)
        # 归一化
        for origin in net.zones:
            for destination in net.zones:
                for t in range(self.transfer_time, -1, -1):
                    if origin != destination:
                        lst = self.FNS_FLS_all[(origin, destination, t)]
                        FLS_subsets = [subset for i in range(1, len(lst[0]) + 1) for subset in itertools.combinations(lst[0], i)]
                        sum_1 = sum(self.decision_porpotion[0][(origin, destination, t, FLS_subset)] for FLS_subset in FLS_subsets)
                        for FLS_subset in FLS_subsets:
                            self.decision_porpotion[0][(origin, destination, t, FLS_subset)] /= sum_1
                        for line in lst[0]:
                            sum_2 = sum(self.decision_porpotion[1][(origin, destination, t, line, node)] for node in lst[1][line])
                            for node in lst[1][line]:
                                self.decision_porpotion[1][(origin, destination, t, line, node)] /= sum_2

class MSA_LNS():
    def __init__(self, sigma, max_iter):
        self.sigma = sigma
        self.max_iter = max_iter
        self.aux_propotion = None
        self.gaps = []

    def Method_of_Successive_Algorithm(self, net, transfer_time, lns):
        # 初始化
        lns.init_FNS_FLS_all(net)
        lns.init_decision_porpotion(net)
        lns.calculate_flow(net)
        # 迭代
        for i in range(self.max_iter + 1):
            lns.efreq_calculation(net)
            print(lns.v[2])
            lns.ANS_ALS_calculation(net)
            # 1. 判断收敛
            if self.convergence_judge(net, lns):
                print(i)
                break
            # 3. 计算辅助比例
            self.aux_propotion_calculation(net, lns)
            # 4. 更新决策比例
            lns.decision_porpotion_update(net, self.aux_propotion, i)
            # 5. 更新流量
            lns.calculate_flow(net)

    def aux_propotion_calculation(self, net, lns):
        aux_propotion_line = {}
        aux_propotion_node = {}
        for origin in net.zones:
            for destination in net.zones:
                for t in range(lns.transfer_time, -1, -1):
                    if origin != destination:
                        lst = lns.FNS_FLS_all[(origin, destination, t)]
                        FLS_subsets = [subset for i in range(1, len(lst[0]) + 1) for subset in itertools.combinations(lst[0], i)]
                        for FLS_subset in FLS_subsets:
                            if list(FLS_subset) == lns.ANS_ALS_all[1][(origin, destination, t)]:
                                aux_propotion_line[(origin, destination, t, FLS_subset)] = 1
                            else:
                                aux_propotion_line[(origin, destination, t, FLS_subset)] = 0
                        for line in lst[0]:
                            for node in lst[1][line]:
                                if node in lns.ANS_ALS_all[0][(origin, destination, t)][line]:
                                    aux_propotion_node[(origin, destination, t, line, node)] = 1
                                else:
                                    aux_propotion_node[(origin, destination, t, line, node)] = 0
        self.aux_propotion = (aux_propotion_line, aux_propotion_node)

    def convergence_judge(self, net, lns):
        term_1 = 0
        term_2 = 0
        term_3 = 0
        term_4 = 0
        for origin in net.zones:
            for destination in net.zones:
                for t in range(lns.transfer_time, -1, -1):
                    if origin != destination:
                        lst = lns.FNS_FLS_all[(origin, destination, t)]
                        FLS_subsets = [subset for i in range(1, len(lst[0]) + 1) for subset in itertools.combinations(lst[0], i)]
                        for FLS_subset in FLS_subsets:
                            term_1 += abs(lns.decision_porpotion[0][(origin, destination, t, FLS_subset)] * (lns.c_ttt_set[(origin, destination, t, FLS_subset)] - lns.min_c_ttt[(origin, destination, t)]))
                            term_2 += lns.decision_porpotion[0][(origin, destination, t, FLS_subset)] * lns.c_ttt_set[(origin, destination, t, FLS_subset)]
                        for line in lst[0]:
                            for node in lst[1][line]:
                                term_3 += abs(lns.decision_porpotion[1][(origin, destination, t, line, node)] * (lns.c_inv_set[(origin, destination, t, line, node)] - lns.min_c_inv[(origin, destination, t, line)]))
                                term_4 += lns.decision_porpotion[1][(origin, destination, t, line, node)] * lns.c_inv_set[(origin, destination, t, line, node)]
        gap = term_1 / term_2 + term_3 / term_4
        print(gap)
        self.gaps.append(gap)
        if gap < self.sigma:
            return True
        else:
            return False

net = Network()
lns = LNS(2)
msa_lns = MSA_LNS(0.001, 100)
msa_lns.Method_of_Successive_Algorithm(net, 3, lns)
print(lns.decision_flow)
print('\n')
print(lns.decision_porpotion)
print('\n')
print(lns.v[2])
print('\n')
# 绘制收敛曲线
plt.plot(msa_lns.gaps)
plt.yscale('log')
plt.show()
print('\n')
