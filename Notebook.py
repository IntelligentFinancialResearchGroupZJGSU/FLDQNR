# -*- coding: UTF-8 -*-
from random import shuffle
import json
import numpy as np
import pandas as pd
from collections import Counter

"""统计每个数据集中各个分数出现的次数"""
# """读入评分数据 ml_1m"""
# """These files contain 1,000,209 anonymous ratings of approximately 3,900 movies
# made by 6,040 MovieLens users who joined MovieLens in 2000. Ratings: [1,5]"""
# Original_Ratings = pd.read_table("DataSet/ratings.dat", sep = '::', header = None, engine = 'python')
# Original_Ratings = Original_Ratings.sort_values(by = 3)  # 将评分数据根据时间戳排序
# Original_Ratings = np.array(Original_Ratings)



"""firmtrust"""
# Original_Ratings = pd.read_table("data/filmtrust/ratings.txt", sep = ' ', header = None, engine = 'python')
# Sequence = list(range(0, len(Original_Ratings)))
# Original_Ratings = np.array(Original_Ratings)
# Original_Ratings = Original_Ratings[Sequence, :]

# """ml_la"""
# Original_Ratings = pd.read_csv("data/ml-la/ratings.csv", header = None)
# Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
# Original_Ratings = np.array(Original_Ratings)  # 原始评分数据
# count = Counter(list(Original_Ratings[:,2]))
# print(len(Original_Ratings))
# print(count)
# for i in range(len(count)):
#     Ratio = count[i + 1] / len(Original_Ratings)
#     print(Ratio)

"""Automotive"""
Original_Ratings = pd.read_csv("data/transratings_Automotive.csv", header = None)
Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
Original_Ratings = np.array(Original_Ratings)
count = Counter(list(Original_Ratings[:,2]))
print(len(Original_Ratings))
print(count)



# data = [[1,2,3,4,5],[2,3,4,5,4],[2,3,4]]
#
# with open("data.json","w") as file:
#     json.dump(data,file)
#
# with open("data.json",'r') as file:
#     tmp = json.load(file)
# print(tmp)


# """随机分组"""
# list = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# shuffle(list)
# n = 3
# m = int(len(list)/n)
# list2 = []
# for i in range(0,len(list),m):
#     list2.append(list[i:i+m])
# print(list2)

# """lambda函数用法"""
# a = 10
# f = lambda x: x * a
# print(f)
# print(type(f))
# print(f(3))
import numpy as np

# a = np.zeros((2,5))
# b = np.ones((2,3))
# print(a)
# print(b)
# c = np.hstack((a,b))
# print(c)

# tmp = list(range(1,100))
# print(tmp)

# temp = [1,2,3,4,5,6]
# user = 3
# if user in temp:
#     pass


# num = 2
# for i in range(num):
#     print(i)
