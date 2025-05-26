import sys
import json
sys.path.append('D:\MyCode\Python\PLM_Transit_Assignment_model')
from PLM.Source_Code.MSA_with_detection import PLM, MSA, NETWORK
import matplotlib.pyplot as plt
import pickle

with open("PLM_Transit_Assignment_model/PLM/Winnipeg_Network/examples/net_low.json", "r") as f:
    net_ = json.load(f)
net = NETWORK(net_, 3)
plm = PLM()
MSA.variance_threshold = 0.0001
MSA.ratio_update = 0.2
MSA.coefficient_requirement = 0.9
msa = MSA(0.0000000001, 50)
time_list = []
updat_num = []
# msa.Self_regulated_averaging_method_Algorithm(net, plm, 0, updat_num, time_list)
# msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, updat_num, time_list)
# msa.Method_of_Successive_Algorithm(net, plm, 0, updat_num, time_list)
# msa.Method_of_Successive_Algorithm(net, plm, 1, updat_num, time_list)
print(updat_num)
plt.plot(range(len(msa.gaps)), msa.gaps, color='b', marker='o', label='gap')
plt.legend()
plt.yscale('log')
plt.xlabel('iteration_num')
plt.ylabel('gap(log)')
plt.title('Convergence curve')
plt.show()

# 保留gap数据与时间数据
# with open("MSA_low.pkl", "wb") as f:
#     pickle.dump((msa_plm.gaps, time_list), f)
