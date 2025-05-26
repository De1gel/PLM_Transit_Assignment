import sys
sys.path.append('D:\MyCode\Python\PLM_Transit_Assignment_model')
from PLM.Source_Code.MSA_with_detection import PLM, MSA, NETWORK
import matplotlib.pyplot as plt

MSA.variance_threshold = 0.001  # 设置方差阈值
colors = ['m', 'r', 'g', 'b', 'y']
markers = ['o', 's', 'D', '^', 'v']

# 图6——efff系数参数从1-6纵向比较
for i in range(1, 6):
    update_num = []
    net = NETWORK(None, i)
    plm = PLM()
    msa = MSA(0.000001, 100)
    msa.Self_regulated_averaging_method_Algorithm(net, plm, 0, update_num, [])  # 1表示使用自相关加速 0表示原算法
    print(fr"alpha={i}\;\beta={net.beta_line['L1']}下流量分配结果")
    print(plm.v)
    # 绘制收敛曲线
    plt.plot(msa.gaps, marker=markers[i - 1], linestyle='-', color=colors[i - 1],
             label=fr'$\alpha={i}\;\beta={net.beta_line["L1"]}$', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
plt.rcParams['legend.fontsize'] = 8
plt.legend()
plt.yscale('log')
plt.xlabel('iteration_num')
plt.ylabel('gap(log)')
plt.title('Convergence curve')
# plt.savefig('sra_diff_alpha.jpg', dpi=1200)
plt.show()

# 图6——不同参数下SRAD与SRA对比
update_num = []
net = NETWORK(None, 1)  # alpha = 1, beta = 6
plm = PLM()
msa = MSA(0.000001, 100)
msa.Self_regulated_averaging_method_Algorithm(net, plm, 0, update_num, [])
plt.plot(msa.gaps, marker='o', linestyle='-', color='m',
         label='SRA', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
update_num = []
plm = PLM()
msa = MSA(0.000001, 100)
msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, update_num, [])
plt.plot(msa.gaps, marker='s', linestyle='-', color='r',
         label='SRAD', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
plt.rcParams['legend.fontsize'] = 8
plt.legend()
plt.yscale('log')
plt.xlabel('iteration_num')
plt.ylabel('gap(log)')
plt.title('Convergence curve')
# plt.savefig('srad_sra_1_6.jpg', dpi=1200)
plt.show()

update_num = []
net = NETWORK(None, 5)  # alpha = 5, beta = 13
plm = PLM()
msa = MSA(0.000001, 100)
msa.Self_regulated_averaging_method_Algorithm(net, plm, 0, update_num, [])
plt.plot(msa.gaps, marker='o', linestyle='-', color='m',
         label='SRA', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
update_num = []
plm = PLM()
msa = MSA(0.000001, 100)
msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, update_num, [])
plt.plot(msa.gaps, marker='s', linestyle='-', color='r',
         label='SRAD', linewidth=0.5, markersize=3)  # 设置线宽为0.5，点大小为5
plt.rcParams['legend.fontsize'] = 8
plt.legend()
plt.yscale('log')
plt.xlabel('iteration_num')
plt.ylabel('gap(log)')
plt.title('Convergence curve')
# plt.savefig('srad_sra_5_13.jpg', dpi=1200)
plt.show()