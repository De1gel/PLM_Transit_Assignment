import sys
sys.path.append('D:\MyCode\Python\PLM_Transit_Assignment_model')
from PLM.Source_Code.MSA_with_detection import PLM, MSA, NETWORK
import matplotlib.pyplot as plt

colors = ['m', 'r', 'g', 'b', 'y']
markers = ['o', 's', 'D', '^', 'v']
net = NETWORK(None, 5)  # 比较该参数下的不同ODA参数的结果
# ODA包括三个参数：queue length, variance threshold, and coefficient requirement
# 比较队列长度
MSA.queue_length = 4
MSA.variance_threshold = 0.01
MSA.coefficient_requirement = 0.99
for i in range(3):
    plm = PLM()
    msa = MSA(0.000001, 100)
    msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, [], [])
    plt.plot(msa.gaps, marker=markers[i], linestyle='-', color=colors[i],
             label=f'$e={MSA.queue_length}$', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
    MSA.queue_length += 2
plt.yscale('log')
plt.xlabel('iteration_num')
plt.ylabel('gap(log)')
plt.title('SRA with different queue length $e$')
plt.legend()
plt.savefig('公交分配/experiment_2/srad_diff_alpha.jpg', dpi=1200)
plt.show()

# 比较方差阈值
MSA.queue_length = 4
MSA.variance_threshold = 0.1
MSA.coefficient_requirement = 0.99
for i in range(3):
    plm = PLM()
    msa = MSA(0.000001, 100)
    msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, [], [])
    plt.plot(msa.gaps, marker=markers[i], linestyle='-', color=colors[i],
             label=f'$\kappa={MSA.variance_threshold}$', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
    MSA.variance_threshold /= 10
plt.yscale('log')
plt.xlabel('iteration_num')
plt.ylabel('gap(log)')
plt.title('SRA with different variance threshold $\kappa$')
plt.legend()
plt.savefig('公交分配/experiment_2/srad_diff_variance.jpg', dpi=1200)
plt.show()

# 比较系数要求
MSA.queue_length = 4
MSA.variance_threshold = 0.001
MSA.coefficient_requirement = 0.9
for i in range(3):
    plm = PLM()
    msa = MSA(0.000001, 100)
    msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, [], [])
    plt.plot(msa.gaps, marker=markers[i], linestyle='-', color=colors[i],
             label=f'$\phi={MSA.coefficient_requirement}$', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
    MSA.coefficient_requirement = 1 - (1 - MSA.coefficient_requirement) / 10
plt.yscale('log')
plt.xlabel('iteration_num')
plt.ylabel('gap(log)')
plt.title('SRA with different coefficient requirement $\phi$')
plt.legend()
plt.savefig('公交分配/experiment_2/srad_diff_coefficient.jpg', dpi=1200)
plt.show()
