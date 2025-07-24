import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# 读取药物相似矩阵
drug_sim_path = "D:\\13645583245\\文献代码\\DRNCL\\dataset\\Ldataset\\lagcn\\drug_sim.csv"
drug_sim_df = pd.read_csv(drug_sim_path, header=None)
# drug_sim_matrix = drug_sim_df.values
# print("药物相似矩阵:")
# print(drug_sim_matrix.shape)

# 读取药物疾病关联矩阵
drug_dis_path = "D:\\13645583245\\文献代码\\DRNCL\\dataset\\Ldataset\\lagcn\\drug_dis.csv"
drug_dis_df = pd.read_csv(drug_dis_path, header=None)
drug_dis_matrix = drug_dis_df.values
print(f"药物疾病关联矩阵:{drug_dis_matrix}")
print(drug_dis_matrix.shape)

# # 读取疾病相似矩阵
# dis_sim_path = "D:\\13645583245\\文献代码\\DRNCL\\dataset\\Ldataset\\lagcn\\dis_sim.csv"
# dis_sim_df = pd.read_csv(dis_sim_path, header=None)
# dis_sim_matrix = dis_sim_df.values
# print("疾病相似矩阵:")
# print(dis_sim_matrix)