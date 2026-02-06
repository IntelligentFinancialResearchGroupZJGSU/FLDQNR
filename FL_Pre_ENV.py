# -*- coding: UTF-8 -*-
# Author: WuTong
# Data: 2022.08.07

"""A3C预测评分的环境"""

import gym
import math
import random
import pandas as pd
from gym.utils import seeding
import numpy as np
from gym import spaces
from DataProcess import *


class FL_ENV(gym.Env):
    def __init__(self,User_Group,User_ID,Movie_ID,K_neighbors,K_items,Max_Rating,Rating_Matrix_Train,DataSetTrain_Rating,Rating_Matrix_Test,DataSetTest_Rating,Original_Ratings):
        self.count = 0
        # 每一个用户的评分由1-Max_Rating表示
        alist = [1] * int(Max_Rating)  # 0表示1，Max_Rating表示最高分
        self.User_ID = User_ID.copy()
        self.Movie_ID = Movie_ID.copy()
        self.n_actions = len(alist)
        self.Rating_Matrix_Train = Rating_Matrix_Train.copy()
        self.DataSetTrain_Rating = DataSetTrain_Rating.copy()
        self.Rating_Matrix_Test = Rating_Matrix_Test.copy()
        self.DataSetTest_Rating = DataSetTest_Rating.copy()
        self.Original_Ratings = Original_Ratings.copy()
        self.K_neighbors = K_neighbors
        self.K_items = K_items
        self.User_Group = User_Group.copy()
        self.Max_Rating = Max_Rating


        # """未归一化的状态空间"""
        # low_high = np.ones((1,K_neighbors + K_neighbors))[0] * Max_Rating # 状态上界
        # low_low = np.ones((1,K_neighbors + K_items)) [0] # 状态下界

        """归一化的状态空间"""
        low_high = np.ones((1, K_neighbors + K_items))[0]  # 状态上界 [0, 1]
        low_low = np.zeros((1, K_neighbors + K_items))[0]  # 状态下界 [0, 1]

        """观察到的状态空间"""
        self.observation_space = spaces.Box(low = low_low, high = low_high)
        # print(self.observation_space)
        self.seed()
        self.state = None
        self.label = None

    """没用  但是得有"""
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def resetenv(self,train_data):  # 初始化agent
        self.state = train_data.copy()  # 随机初始化缓存状态和可用缓存

        return np.array(self.state)


    def reset_test(self,test_data, test_label):
        state_label = np.hstack((test_data,test_label.T))

        return np.array(self.state)

    def step_agent(self,action,label,icount,max_step,Max_Rating):
        """以预测出的评分与真实评分的负的误差绝对值作为奖励"""
        if icount + 1 == max_step:
            done = True
        else:
            done = False
        # 4. 改进奖励函数：使用更平滑的奖励设计，对小误差给予更好的奖励
        # action是0到Max_Rating-1，label是1到Max_Rating
        error = abs((action + 1) - label)
        # 使用更平滑的奖励：对小误差给予更好的奖励，对大误差惩罚更重
        if error == 0:
            reward = 0.0  # 完全正确，无惩罚
        elif error <= 1.0:
            reward = -error / Max_Rating  # 小误差：线性惩罚
        else:
            reward = -(1.0 + (error - 1.0) ** 2) / Max_Rating  # 大误差：平方惩罚
        reward = reward * Max_Rating  # 放大到合理范围
        # reward = -abs(action + 1  - label)  # 这是正常的奖励
        # reward = -abs(action  - label * Max_Rating )  # 这是正则化的奖励
        # if ep_t + 1 < MAX_EP_STEP:
        #     done = False
        #     tmp = Sequence[ep_t + 1]
        #     s_ = state[tmp]
        # else:
        #     s_ = np.zeros((1,len(state)))[0]
        info = {}

        # print("Accuracy:", i_count / MAX_EP_STEP)  # 打印准确率

        return reward, done, info


    def step_pre(self,predictdata,label):
        """以预测出的评分与真实评分的负的误差绝对值作为奖励"""
        reward = - abs(predictdata + 1 - label)


        return reward


    def step_RMSE(self, action,predictdata,ep_t,trainlabel):
        """以预测出的评分与真实评分的负的误差绝对值作为奖励"""
        a = round(float(action))

        # if a == trainlabel[ep_t]:
        # reward = pow((a-trainlabel[ep_t])^2,2)
        MAX_EP_STEP = len(trainlabel)
        if ep_t + 1 <MAX_EP_STEP:
            s_ = predictdata[ep_t, :]
        else:
            s_ = []

        # print("Accuracy:", i_count / MAX_EP_STEP)  # 打印准确率

        return s_ ,a, {}




