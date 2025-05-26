import numpy as np
import itertools
import copy
import queue
import matplotlib.pyplot as plt
import pickle
import time
import pdb
import math

class NETWORK():
    def __init__(self, net_, alpha):
        if net_ is None:
            self.zones = ('A', 'X', 'Y', 'B')
            self.lines = ('L1', 'L2', 'L3', 'L4')
            self.alpha_line = {line: alpha for line in self.lines}
            self.beta_line = {line: round(math.log(0.2 / (self.alpha_line[line] * (1 - 0.2))) / math.log(0.8)) for line in self.lines}
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
            self.nodes_lineset = self.init_nodes_lineset()
            # 初始化OD需求
            self.demand = {
                'A': {'B': 1600},
                'X': {'B': 1300}
            }
            self.destination = self.get_destination()
        else:
            self.zones = tuple(net_['zones'])
            self.lines = tuple(net_['lines'])
            self.alpha_line = {line: alpha for line in self.lines}
            self.beta_line = {line: round(math.log(0.2 / (self.alpha_line[line] * (1 - 0.2))) / math.log(0.8)) for line in self.lines}
            self.scheduled_frequency = net_['scheduled_frequency']
            self.vehicle_capacity = net_['vehicle_capacity']
            self.itinerary = net_['itinerary']
            self.itinerary_matrix = self.init_itinerary_matrix()
            self.nodes_lineset = self.init_nodes_lineset()
            # 初始化OD需求
            self.demand = net_['demand']
            self.destination = self.get_destination()

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

    def init_nodes_lineset(self):
        nodes_lineset = {}
        for origin in self.zones:
            lines_before = ['Lp']
            lines_after = []
            for line in self.lines:
                for node in self.itinerary[line][0][1:]:
                    if node == origin:
                        lines_before.append(line)
                        break
                for node in self.itinerary[line][0][:-1]:
                    if node == origin:
                        lines_after.append(line)
                        break
            nodes_lineset[origin] = (lines_before, lines_after)
        return nodes_lineset

    def get_destination(self):
        # 计算每个节点的目的地
        destination = []
        for origin in self.demand.keys():
            for node in self.demand[origin].keys():
                if node not in destination:
                    destination.append(node)
        return destination

class PLM():
    def __init__(self):
        self.decision_porpotion = None
        self.decision_flow = None
        self.efreq_line = None
        self.c_ttt_set = None
        self.min_c_ttt = None
        self.c_inv_set = None
        self.ALS = None
        self.v = None
        self.distance = 0
        self.h_k = 0

    def init_decision_porpotion(self, net):
        Lambda_line = {}
        for origin in net.zones:
            for destination in net.destination:
                for line_before in net.nodes_lineset[origin][0]:
                    if origin != destination:
                        Lambda_line[(origin, destination, line_before)] = {}
                        lst = net.nodes_lineset[origin][1]
                        if lst == []:
                            continue
                        FLS_subsets = [subset for i in range(1, len(lst) + 1) for subset in itertools.combinations(lst, i)]
                        for FLS_subset in FLS_subsets:
                            # 全有全无
                            if FLS_subset == FLS_subsets[0]:
                                Lambda_line[(origin, destination, line_before)][tuple(sorted(FLS_subset))] = 1
                            else:
                                Lambda_line[(origin, destination, line_before)][tuple(sorted(FLS_subset))] = 0
        self.decision_porpotion = Lambda_line
        self.efreq_line = {}
        for line in net.lines:
            for node in net.itinerary[line][0]:
                for line_before in net.nodes_lineset[node][0]:
                    self.efreq_line[(line, node, line_before)] = net.scheduled_frequency[line]

    def calculate_flow(self, net):
        def Load_flow(origin, destination, demand, line_before, node_acc=[]):
            if origin in node_acc or origin == destination:
                return
            else:
                node_acc_local = node_acc + [origin]
            flow_local = {line: 0 for line in net.nodes_lineset[origin][1]}
            for strategy in self.decision_porpotion[(origin, destination, line_before)].keys():
                if self.decision_porpotion[(origin, destination, line_before)][strategy] > 0:
                    flow_strategy = demand * self.decision_porpotion[(origin, destination, line_before)][strategy]
                    efff_sum = sum(self.efreq_line[(line, origin, line_before)] for line in strategy)
                    for line in strategy:
                        flow_local[line] += flow_strategy * self.efreq_line[(line, origin, line_before)] / efff_sum
            for line in net.nodes_lineset[origin][1]:
                if flow_local[line] > 0:
                    node_next = net.itinerary[line][0][net.itinerary[line][0].index(origin) + 1]
                    flow_line_tmp[(origin, destination, line)] += flow_local[line]
                    Load_flow(node_next, destination, flow_local[line], line, node_acc_local)

        flow_line = {(origin, destination, line): 0 for origin in net.zones for destination in net.zones if origin != destination for line in net.nodes_lineset[origin][1]}
        flow_line_tmp = {(origin, destination, line): 0 for origin in net.zones for destination in net.zones if origin != destination for line in net.nodes_lineset[origin][1]}
        for origin in net.demand:
            for destination in net.demand[origin]:
                Load_flow(origin, destination, net.demand[origin][destination], 'Lp', [])
                for key in flow_line_tmp.keys():
                    flow_line[key] += flow_line_tmp[key]
                flow_line_tmp = {(origin, destination, line): 0 for origin in net.zones for destination in net.zones if origin != destination for line in net.nodes_lineset[origin][1]}
        self.decision_flow = flow_line

    def efreq_calculation(self, net):
        def eff_function(v, line):
            Numerator = net.scheduled_frequency[line] * 1
            saturation_rate = v / (net.scheduled_frequency[line] * net.vehicle_capacity[line])
            Denumenator = 1 + net.alpha_line[line] * saturation_rate ** net.beta_line[line]
            return Numerator / Denumenator
        v_0 = {}
        efreq_line = {}
        for node in net.zones:
            for line in net.nodes_lineset[node][1]:
                v_0[(line, node)] = 0
                for node_ in net.zones:
                    if node != node_:
                        v_0[(line, node)] += self.decision_flow[(node, node_, line)]
                for line_before in net.nodes_lineset[node][0]:
                    if line_before == line:
                        efreq_line[(line, node, line_before)] = 120
                    else:
                        efreq_line[(line, node, line_before)] = eff_function(v_0[(line, node)], line)
                self.efreq_line = efreq_line
        self.v = v_0

    def ALS_calculation(self, net):
        def c_ttt_calculation(origin, destination, line_before, node_acc=[]):
            min_c_ttt[(origin, destination, line_before)] = 99999
            ALS[(origin, destination, line_before)] = [net.nodes_lineset[origin][1][0]] if net.nodes_lineset[origin][1] != [] else []
            if origin in node_acc:
                return 99999
            else:
                node_acc_local = node_acc + [origin]
            if origin == destination:
                min_c_ttt[(origin, destination, line_before)] = 0
                return 0
            lst = net.nodes_lineset[origin][1]
            FLS_subsets = [subset for i in range(1, len(lst) + 1) for subset in itertools.combinations(lst, i)]
            for lineset in FLS_subsets:
                lineset = tuple(sorted(lineset))
                # 计算有效发车频率
                efreq = sum(self.efreq_line[(line, origin, line_before)] for line in lineset) / 60
                # 计算在车时间
                exp_inv_time = 0
                for line in lineset:
                    node_next = net.itinerary[line][0][net.itinerary[line][0].index(origin) + 1]
                    if (node_next, destination, line) not in min_c_ttt.keys():
                        c_ttt_calculation(node_next, destination, line, copy.deepcopy(node_acc_local))
                    c_inv_set[(origin, destination, line_before, line)] = min_c_ttt[(node_next, destination, line)] + net.itinerary[line][1][net.itinerary[line][0].index(origin)]
                    exp_inv_time += c_inv_set[(origin, destination, line_before, line)] * self.efreq_line[(line, origin, line_before)] / 60
                # 计算期望行程时间
                c_ttt = (1 + exp_inv_time) / efreq
                c_ttt_set[(origin, destination, line_before, lineset)] = c_ttt
                if c_ttt < min_c_ttt[(origin, destination, line_before)]:
                    min_c_ttt[(origin, destination, line_before)] = c_ttt
                    ALS[(origin, destination, line_before)] = list(lineset)
            return min_c_ttt[(origin, destination, line_before)]
        c_ttt_set = {}
        c_inv_set = {}
        min_c_ttt = {}
        ALS = {}
        for origin in net.zones:
            for destination in net.destination:
                for line_before in net.nodes_lineset[origin][0]:
                    c_ttt_calculation(origin, destination, line_before, [])
        self.c_ttt_set = c_ttt_set
        self.c_inv_set = c_inv_set
        self.min_c_ttt = min_c_ttt
        self.ALS = ALS

    def decision_porpotion_update(self, aux_propotion, i):
        for key_1 in self.decision_porpotion.keys():
            sum = 0
            for key_2 in self.decision_porpotion[key_1].keys():
                self.decision_porpotion[key_1][key_2] = self.decision_porpotion[key_1][key_2] + (aux_propotion[key_1][key_2] - self.decision_porpotion[key_1][key_2]) / (i + 1)
                sum += self.decision_porpotion[key_1][key_2]
            for key_2 in self.decision_porpotion[key_1].keys():
                self.decision_porpotion[key_1][key_2] /= sum

    def decision_porpotion_update_SRA(self, aux_propotion, i):
        # 求解最优解与辅助解的距离
        dec_vals = []
        aux_vals = []
        for k1 in self.decision_porpotion:
            for k2 in self.decision_porpotion[k1]:
                dec_vals.append(self.decision_porpotion[k1][k2])
                aux_vals.append(aux_propotion[k1][k2])
        distance_ = np.linalg.norm(np.array(dec_vals) - np.array(aux_vals), ord=2)
        if i == 0:
            self.h_k = 1
        elif distance_ < self.distance:
            self.h_k = self.h_k + 0.5
        else:
            self.h_k = self.h_k + 1.5
        self.distance = distance_
        for key_1 in self.decision_porpotion.keys():
            sum = 0
            for key_2 in self.decision_porpotion[key_1].keys():
                self.decision_porpotion[key_1][key_2] = self.decision_porpotion[key_1][key_2] + (aux_propotion[key_1][key_2] - self.decision_porpotion[key_1][key_2]) / self.h_k
                sum += self.decision_porpotion[key_1][key_2]
            for key_2 in self.decision_porpotion[key_1].keys():
                self.decision_porpotion[key_1][key_2] /= sum

class MSA():
    queue_length = 4  # 队列长度
    variance_threshold = 0.01  # 方差阈值
    coefficient_requirement = 0.99  # 系数要求
    ratio_update = 0.05  # 更新比例

    def __init__(self, sigma, max_iter):
        self.sigma = sigma
        self.max_iter = max_iter
        self.aux_propotion = None
        self.gaps = []

    def Method_of_Successive_Algorithm(self, net, plm, flag, updat_num, time_list):
        # 初始化
        eff = queue.Queue()
        plm.init_decision_porpotion(net)
        print('initilization completed...')
        iter_num = 0
        # 迭代
        for i in range(self.max_iter):
            print(i)
            plm.calculate_flow(net)
            plm.efreq_calculation(net)
            # print('flow calculation completed...')
            if flag:
                eff.put(plm.efreq_line)
                if self.Autocorrelation_acceleration(plm, eff, updat_num):
                    plm.calculate_flow(net)
                    plm.efreq_calculation(net)
                    eff.put(plm.efreq_line)
            # print(plm.v)
            # print('cost calculating...')
            plm.ALS_calculation(net)
            time_list.append(time.process_time())
            if self.convergence_judge(plm):
                print(i)
                break
            # 3. 计算辅助比例
            # print('updating...')
            self.aux_propotion_calculation(plm)
            # 4. 更新决策比例
            plm.decision_porpotion_update(self.aux_propotion, iter_num)
            iter_num += 1

    def Self_regulated_averaging_method_Algorithm(self, net, plm, flag, updat_num, time_list):
        # 初始化
        eff = queue.Queue()
        plm.init_decision_porpotion(net)
        print('initilization completed...')
        iter_num = 0
        # 迭代
        for i in range(self.max_iter):
            print(i)
            plm.calculate_flow(net)
            plm.efreq_calculation(net)
            if flag:
                eff.put(plm.efreq_line)
                if self.Autocorrelation_acceleration(plm, eff, updat_num):
                    plm.calculate_flow(net)
                    plm.efreq_calculation(net)
                    eff.put(plm.efreq_line)
            # print(plm.v)
            plm.ALS_calculation(net)
            time_list.append(time.process_time())
            if self.convergence_judge(plm):
                print(i)
                break
            # 3. 计算辅助比例
            self.aux_propotion_calculation(plm)
            # 4. 更新决策比例
            plm.decision_porpotion_update_SRA(self.aux_propotion, iter_num)
            iter_num += 1

    def Autocorrelation_acceleration(self, plm, eff, updat_num):
        while eff.qsize() > self.queue_length:
            eff.get()
        if eff.qsize() == self.queue_length:
            return self.ifautocorrelation(plm, eff, updat_num)
        else:
            return 0

    def ifautocorrelation(self, plm, eff, updat_num):
        updat_num_ = 0

        # 自相关系数
        def autocorrelation(x):
            n = len(x)
            result = np.correlate(x, x, mode='full')
            result = result[result.size // 2:]
            result /= (n - np.arange(n))  # 归一化
            return result[2] / result[0]  # 标准化
        eff_list = list(eff.queue)
        eff_list_ = [list(eff.values()) for eff in eff_list]
        eff_list_ = np.array(eff_list_)
        eff_list_ = eff_list_.T
        keys = list(eff_list[0].keys())
        for key, eff_ in zip(keys, eff_list_):
            if autocorrelation(eff_) > self.coefficient_requirement:
                if np.var(eff_) > self.variance_threshold:
                    plm.efreq_line[key] = np.mean(eff_)
                    updat_num_ += 1
        updat_num.append(updat_num_)
        if updat_num_ > self.ratio_update * len(keys):
            return 1
        else:
            return 0

    def aux_propotion_calculation(self, plm):
        aux_propotion_line = {}
        for key_1 in plm.decision_porpotion.keys():
            aux_propotion_line[key_1] = {}
            for key_2 in plm.decision_porpotion[key_1].keys():
                if set(key_2) == set(plm.ALS[(key_1)]):
                    aux_propotion_line[key_1][key_2] = 1
                else:
                    aux_propotion_line[key_1][key_2] = 0
        self.aux_propotion = aux_propotion_line

    def convergence_judge(self, plm):
        term_1 = 0
        term_2 = 0
        for key_1 in plm.decision_porpotion.keys():
            for key_2 in plm.decision_porpotion[key_1].keys():
                key_3 = key_1 + (key_2,)
                if plm.min_c_ttt[key_1] == 99999:
                    continue
                term_1 += abs(plm.decision_porpotion[key_1][key_2] * (plm.c_ttt_set[key_3] - plm.min_c_ttt[key_1]))
                term_2 += plm.decision_porpotion[key_1][key_2] * plm.c_ttt_set[key_3]
        gap = term_1 / term_2
        print(gap)
        self.gaps.append(gap)
        if gap < self.sigma:
            return True
        else:
            return False

if __name__ == '__main__':
    net = NETWORK(None, 1)  # alpha = 1, beta = 6
    plm = PLM()
    msa = MSA(0.000001, 100)
    update_num = []
    time_list = []
    msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, update_num, time_list)  # 1表示使用自相关加速 0表示原算法
    print(fr'alpha={net.alpha_line["L1"]}\;\beta={net.beta_line["L1"]}下流量分配结果')
    print(plm.v)
    # 绘制收敛曲线
    plt.plot(msa.gaps, marker='o', linestyle='-', color='m',
             label=fr'$\alpha={net.alpha_line["L1"]}\;\beta={net.beta_line["L1"]}$', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
    plt.rcParams['legend.fontsize'] = 8
    plt.legend()
    plt.yscale('log')
    plt.xlabel('iteration_num')
    plt.ylabel('gap(log)')
    plt.title('Convergence curve')
    plt.show()
