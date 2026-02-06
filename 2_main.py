# -*- coding: UTF-8 -*-
# Author: Wu Tong
# Date: 2023.7.6

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from FL_Pre_ENV import *

from pytorch.Federator import Federator
from pytorch.DQN import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer
from DataProcess import *




"""此处是训练阶段的设置"""
n_episode = 20  # episode数量
update_rate = 300
K_neighbors = 5  # 定义协同过滤的邻居用户数量
K_items = 5 # 定义协同过滤的邻居内容数量

# 数据集列表
datasets = ["filmtrust", "ml_1m", "ml_la", "automovie"]

# 训练集比例列表
train_ratios = [0.25, 0.5, 0.75]

# 存储结果
results = []

if __name__ == "__main__":
    # 循环处理每个数据集
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        # 循环处理每个训练集比例
        for ratio in train_ratios:
            print(f"\n开始处理数据集 {dataset_name} - 训练集比例: {ratio}")
            
            # 根据数据集和比例确定数据目录（与主函数1保存的目录对应）
            ratio_str = str(ratio).replace('.', '_')
            data_dir = f"train_data_{dataset_name}_{ratio_str}"
            
            # 从对应比例的目录读取数据
            Rating_Matrix_Train = pd.read_csv(f"{data_dir}/Rating_Matrix_Train.csv", index_col = 0)
            Rating_Matrix_Train = np.array(Rating_Matrix_Train)

            Rating_Matrix_Test = pd.read_csv(f"{data_dir}/Rating_Matrix_Test.csv", index_col = 0)
            Rating_Matrix_Test = np.array(Rating_Matrix_Test)

            DataSetTrain_Rating = pd.read_csv(f"{data_dir}/DataSetTrain_Rating.csv", index_col = 0)
            DataSetTrain_Rating = np.array(DataSetTrain_Rating)

            DataSetTest_Rating = pd.read_csv(f"{data_dir}/DataSetTest_Rating.csv", index_col = 0)
            DataSetTest_Rating = np.array(DataSetTest_Rating)

            Original_Ratings = pd.read_csv(f"{data_dir}/Original_Ratings.csv", index_col = 0)
            Original_Ratings = np.array(Original_Ratings)

            """读取用户分组数据"""
            with open(f"{data_dir}/User_Group.txt","r") as file:
                User_Group_lines = file.readlines()

            User_ID = np.array(pd.read_csv(f"{data_dir}/User_ID.csv", index_col = 0))
            Movie_ID = np.array(pd.read_csv(f"{data_dir}/Movie_ID.csv", index_col = 0))

            Num_User = len(User_ID)
            Num_Content = len(Movie_ID)
            Max_Rating = max(DataSetTrain_Rating[:, 2])
            n_agents = len(User_Group_lines)  # 每一个用户作为一个终端

            """生成主网络的环境和参数"""
            args = {
                "env_fn": lambda : FL_ENV(User_Group_lines,User_ID,Movie_ID,K_neighbors,K_items,Max_Rating,Rating_Matrix_Train,DataSetTrain_Rating,Rating_Matrix_Test,DataSetTest_Rating,Original_Ratings),
                "Qnet": FCQ,
                "buffer": ReplayBuffer,

                "net_args": {
                    "hidden_layers": (256, 256),
                    "activation_fn": torch.nn.functional.relu6,
                    "optimizer": torch.optim.Adam,  # torch.optim.RMSprop
                    # 3. 根据训练集比例调整学习率：比例越大，学习率稍微降低，提高稳定性
                    "learning_rate": 0.0002 * (1.0 - ratio * 0.2),  # 0.25->0.00019, 0.5->0.00018, 0.75->0.00017
                },

                "max_epsilon": 1,
                "min_epsilon": 0.05,
                "decay_steps": 80,
                "gamma": 0.99,
                "target_update_rate": 5,  # 5
                "min_buffer": 128
            }

            fed_rewards = np.zeros(n_episode)   # 每一个episode的联邦奖励，可以认为是考虑了所有用户的MAE或者RMSE或者其它的reward
            fed = Federator(n_agents = n_agents, update_rate = update_rate, args = args)
            fed_rewards, rmse, mae = fed.train(n_episode)
            
            # 保存结果
            results.append({
                '数据集': dataset_name,
                '训练集比例': ratio,
                'RMSE': rmse,
                'MAE': mae
            })
            
            # 为每个数据集和比例保存单独的xlsx文件
            ratio_str = str(ratio).replace('.', '_')  # 将0.25转换为0_25
            filename = f"FLDQNR_{dataset_name}_{ratio_str}.xlsx"
            ratio_results_df = pd.DataFrame([{
                '数据集': dataset_name,
                '训练集比例': ratio,
                'RMSE': rmse,
                'MAE': mae
            }])
            ratio_results_df.to_excel(filename, index=False)
            print(f"数据集 {dataset_name} - 训练集比例 {ratio} 完成 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            print(f"结果已保存到 {filename}")
    
    # 保存所有结果到一个汇总文件
    all_results_df = pd.DataFrame(results)
    all_results_df.to_excel("FLDQNR_所有结果.xlsx", index=False)
    print(f"\n所有结果汇总已保存到 FLDQNR_所有结果.xlsx")
    print(all_results_df)

