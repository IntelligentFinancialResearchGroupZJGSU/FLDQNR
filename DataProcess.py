# -*- coding: UTF-8 -*-
import numpy as np
import random
import pandas as pd
import GetKNearestNeighbor
from random import shuffle
import scipy.sparse as sparse
import math

"""计算用户之间的相似度"""
def CalculateSimUser(PreMu,Num_User):
    Sim_ij = np.zeros((Num_User,Num_User))
    for i in range(Num_User):
        for j in range(i+1,Num_User):
            Sim_ij[i,j] = np.dot(PreMu[i],PreMu[j])/(np.linalg.norm(PreMu[i])*np.linalg.norm(PreMu[j]))
            Sim_ij[j,i] = Sim_ij[i,j]
    # """归一化"""
    # for i in range(Num_User):
    #     Sim_ij[i] = Sim_ij[i] / sum(Sim_ij[i])
    return Sim_ij

"""获取数据集中总的类别数"""
def GetCategories(Original_Movies):
    temp = []
    for i in range(len(Original_Movies)):
        tmp = Original_Movies[2][i].split('|')
        temp = [*temp,*tmp]  # 合并列表
    temp = np.unique(temp)
    return temp

"""获得内容关于类别的OneHot矩阵"""
def GetDataOneHot(DataSetTrain_Movie,Category,Movie_ID):
    Category = Category.tolist()
    DataOneHot = np.zeros((len(Movie_ID),len(Category)))
    for i in range(len(Movie_ID)):
        tmp = DataSetTrain_Movie[2][i].split('|')
        for j in range(len(tmp)):
            tmp_index = Category.index(tmp[j])
            DataOneHot[i,tmp_index] = 1
    pass

    return DataOneHot


"""根据预测的分数补全评分矩阵"""
def CompleteRatings(Rating_Matrix,pre):
    icount = 0  # 计数器，统计是第几个预测的分数
    Rating_Matrix_tmp = np.array(Rating_Matrix)
    for i in range(Rating_Matrix_tmp.shape[0]):
        for j in range(Rating_Matrix_tmp.shape[1]):
            if Rating_Matrix_tmp[i,j] == 0:
                Rating_Matrix_tmp[i,j] = pre[icount]
                icount += 1
            else:
                continue
    return Rating_Matrix_tmp

def unique(old_list):
    # Minimize the use_id and item_id in the dataset
    count = 0   ## Update from count = 1
    dic = {}
    for i in range(len(old_list)):
        if old_list[i] in dic:
            old_list[i] = dic[old_list[i]]
        else:
            dic[old_list[i]] = count
            old_list[i] = count
            count += 1
    return old_list

"""将用户分组"""
def Get_User_Group(User_ID,Num_Cluster):
    User_ID_tmp = User_ID[:,1]
    User_ID_tmp = list(User_ID_tmp)
    shuffle(User_ID_tmp)
    m = int(len(User_ID_tmp)/Num_Cluster)
    User_Group = []
    for i in range(0,len(User_ID_tmp),m):
        User_Group.append(User_ID_tmp[i:i+m])
    tmp = User_Group[5]
    tmp1 = User_Group[6]
    User_Group[5] = tmp + tmp1
    del User_Group[6]
    return User_Group

"""对电影数据集进行处理，随机选择给定的用户数和电影数，并分成训练集和测试集"""
def DataProcessMovies(Ratings_Data,User_ID,Movie_ID):
    """选择只包含User_ID和Movie_ID的评分数据集"""
    Ratings = Ratings_Data[:,0:3]

    """生成用户对内容的评分矩阵，用于后续的预测"""
    Rating_Matrix = np.zeros((len(User_ID),len(Movie_ID)))
    Rating_Matrix = pd.DataFrame(Rating_Matrix,index=User_ID[:,0],columns=Movie_ID[:,0])
    for i in range(len(Ratings)):
        tmp_user = Ratings[i, 0]  # 用户ID
        tmp_movie = Ratings[i, 1]  # 电影ID
        Rating_Matrix[tmp_movie][tmp_user] = Ratings[i,2]

    return Ratings, Rating_Matrix

"""生成每一个用户对每一个内容的评分矩阵"""
def readRating(DataSetTrain_Rating,User_ID,Movie_ID):
    mtx_rating = np.zeros((len(User_ID),len(Movie_ID)))
    mtx_rating = pd.DataFrame(mtx_rating)
    for i in range(len(DataSetTrain_Rating[0])):
        mtx_rating[DataSetTrain_Rating[1][i]][DataSetTrain_Rating[0][i]] = DataSetTrain_Rating[2][i]

    return mtx_rating


"""处理数据集，生成输入。其中，输入为N×(k+l)的矩阵"""
def obtain_data(Rating_Matrix_New, Num_User, Num_Content, mtx_np,neighbor_user, k_user,neighbor_item, k_item, User_ID, Movie_ID,MeanUser, MeanContent):
    X_trains, y_trains =[],[]
    mtx_np = np.array(mtx_np)
    y_new = []
    x_new = []
    count = 0

    for id in range(0, Num_User):
        # Import rating information of all neighbors of users into X_np
        X = []
        for neighbor_id in neighbor_user[id]:    # 依次遍历用户id的每一个邻居
            X.append(Rating_Matrix_New[neighbor_id])        # 将用户id的每一个邻居的评价扩展为一个列表
        X_np = np.array(X, dtype=float)
        X_np = np.reshape(X_np, (k_user, Num_Content))   # 第id个用户的相邻用户的评分矩阵

        # Store user rating information
        y = mtx_np[id]  # 第id个用户的评分向量
        y = np.reshape(y, (1, Num_Content))

        # Transpose, each line denotes the rating information of all neighbors
        X_np = X_np.T   # 将X和y转置，则每一行代表所有邻居用户对某一个内容的评分
        y = y.T

        # Pick out items that have been rated by user u  挑选出已经被用户u评价过的内容
        origine_index = []  # Record the original index
        for keys in range(Num_Content):
            if y[keys] != 0:   # 当用户对内容的评分不为0时，抽出该数据
                temp = []   # 用户u的邻居用户对内容keys的评分，这个值只会增加不会改变，用来存储下面的tmp
                tmp = []  # 用户u的邻居用户对内容keys的评分，这个值是临时的
                for k in range((X_np[keys].shape[0])):  # 当用户u对内容keys的评分不为0时，则将u的邻居用户对该内容的评分提取出来（应该会当作训练标签）
                    tmprating = X_np[keys][k]
                    # if(tmprating==0): # 当邻居用户对该内容的评分为0，则用这个用户对该内容的最邻近的评分不为0的内容替代
                    #     tmprating = MeanUser[neighbor_user[id][k]-1]
                    #     # for node_neighbor_item in neighbor_item[keys]:
                    #     #     if (X_np[node_neighbor_item][k] != 0):
                    #     #         tmprating = X_np[node_neighbor_item][k]
                    #     #         break

                    tmp.append(tmprating)
                temp.extend(tmp)
                # Add the rating information of neighbor items
                for neighbor_id in neighbor_item[keys]:  # 增加用户u对内容keys的邻居内容的评价到上面的temp中
                    tmprating = Rating_Matrix_New[id][neighbor_id]
                    # if(mtx_np[id][neighbor_id] == 0):    # 如果用户u对邻居内容的评分为0，则用该用户对邻居内容的邻居评分替代
                    #     tmprating = MeanContent[neighbor_id - 1]
                    #     # for node_neighbor_id in neighbor_item[neighbor_id]:
                    #     #     if(mtx_np[id][node_neighbor_id] != 0):
                    #     #         tmprating = mtx_np[id][node_neighbor_id]
                    #     #         break
                    temp.append(tmprating)
                x_new.append(temp)  # 将用户评分和item评分合并后的新的输入，此处先用列表存储，后面会reshape成输入矩阵
                y_new.extend(y[keys])   # 将用户u对内容keys的评价存储在列表，后续应该会作为训练标签
                origine_index.append(keys)
                count += 1
        # Convert list to array form for easy training
        # Split data into training and test sets

    y_new = np.reshape(y_new, (count, 1))  # 训练标签
    x_new = np.reshape(x_new, (count, k_user + k_item))  # 训练输入

    X = x_new.tolist()  # 训练集输入，用列表表示

    X_tmp = []
    Y_tmp = []
    for i in range(len(X)):
        # user_tmp = []
        # item_tmp = []
        user_tmp = X[i][0:k_user] # 目标用户的邻居用户评分
        item_tmp = X[i][k_user:k_user+k_item]  # 目标用户对目标内容的邻居内容评分
        user_tmp.extend(item_tmp)
        X_tmp.append(user_tmp)

    X_trains.extend(X_tmp)

    y_trains.extend(y_new.tolist())



    X_trainsM = np.reshape(X_trains, (len(X_trains),len(X_trains[0])))

    y_trainsM = np.reshape(y_trains, (len(y_trains), len(y_trains[0])))

    return X_trainsM, y_trainsM

def CF(k_item, k_user, Rating_Matrix):
    """先补全评分矩阵，再中邻居用户"""
    Rating_Matrix = np.array(Rating_Matrix)
    # Rating_Matrix_New = np.zeros((len(Rating_Matrix[:,0]),len(Rating_Matrix[0,:])))

    """这一步是为了解决原始的评分矩阵过于稀疏的问题，此处的意思是用一个用户对所有内容的平均评分来填充其所有缺失的评分"""
    # """生成每一个用户对所有内容评价的平均分"""
    # mean_user = []
    # for i in range(Rating_Matrix.shape[0]):
    #     temp = Rating_Matrix[i]
    #     if len(temp[temp != 0]) == 0:
    #         mean_user.append(0)
    #     else:
    #         mean_user.append(np.mean(temp[temp != 0]))
    #
    # # """生成每一个内容被所有评价过的平均分"""
    # # mean_item = []
    # # for i in range(Rating_Matrix.shape[1]):
    # #     temp = Rating_Matrix.T[i]
    # #     if len(temp[temp != 0]) == 0:
    # #         mean_item.append(0)
    # #     else:
    # #         mean_item.append(np.mean(temp[temp != 0]))  # 根据已有的评价计算平均分数
    #
    # for i in range(Rating_Matrix.shape[0]):
    #     for j in range(Rating_Matrix.shape[1]):
    #         if Rating_Matrix[i][j] != 0:
    #             Rating_Matrix_New[i][j] = Rating_Matrix[i][j]
    #         else:
    #             Rating_Matrix_New[i][j] = mean_user[i]  # 为0的评分就用平均值替代


    neighbor_user = GetKNearestNeighbor.k_neighbors(Rating_Matrix, k_user, len(Rating_Matrix[:,0]))
    """将用户评分转置，变成内容的评分矩阵"""
    Rating_Matrix_Content = Rating_Matrix.T
    neighbor_item = GetKNearestNeighbor.k_neighbors(Rating_Matrix_Content, k_item, len(Rating_Matrix[0,:]))

    # """将评分低于两个的用户去除"""
    # """这一步是为了与baseline作对比，baseline中是将每一个用户的评分分为3:1，但是当用户评分数量小于3时，将没有测试集，因此此处直接删掉这部分数据"""
    # for i in range(Num_User):
    #     if len(np.nonzero(Rating_Matrix[i,:])[0]) <= 2:
    #         Rating_Matrix[i, :] = 0

    return neighbor_user,neighbor_item



def obtain_data(User_ID,Rating_Matrix,DataSet_Rating,User_Neighbor,Item_Neighbor):  # 两个邻居的ID都需要 -1
    inputdata = np.zeros((1,len(User_Neighbor[0]) + len(Item_Neighbor[0])))# 只包含该区域用户的评分数据
    User_Neighbor = np.array(User_Neighbor.copy())
    Item_Neighbor = np.array(Item_Neighbor.copy())
    Mean_User = np.nan_to_num(np.sum(Rating_Matrix,axis = 1) / np.count_nonzero(Rating_Matrix,axis = 1)) # 根据用户已有的评分求出的用户评价平均分
    Mean_Item = np.nan_to_num(np.sum(Rating_Matrix,axis = 0) / np.count_nonzero(Rating_Matrix,axis = 0))
    Rating_Matrix_User = Rating_Matrix.copy()
    Rating_Matrix_Content = Rating_Matrix.copy().T  # 从item角度的评分矩阵
    """将原评分矩阵中为0的元素替换为均分"""
    for i in range(len(Mean_User)):
        Rating_Matrix_User[i] = np.where( Rating_Matrix_User[i] == 0, Mean_User[i], Rating_Matrix_User[i])

    for i in range(len(Mean_Item)):
        Rating_Matrix_Content[i] = np.where(Rating_Matrix_Content[i] == 0, Mean_Item[i], Rating_Matrix_Content[i])

    for i in range(len(DataSet_Rating)):  # 获取协同评分作为输入
        userid = int(DataSet_Rating[i,0])
        itemid = int(DataSet_Rating[i,1])
        tmp1 = Rating_Matrix_User[User_Neighbor[userid-1] - 1,itemid-1]  # 邻居用户对该内容的评分
        tmp2 = Rating_Matrix_Content[Item_Neighbor[itemid-1] - 1, userid-1]  # 该用户对邻居物品的评分
        collaborative_rating = np.hstack((tmp1,tmp2))
        inputdata = np.vstack((inputdata,collaborative_rating))
    label = DataSet_Rating[:,2]
    inputdata = np.delete(inputdata,0,axis = 0)
    return inputdata, label

def obtain_data_evaluate(Rating_Matrix,DataSet_Rating,User_Neighbor,Item_Neighbor):
    inputdata = np.zeros((1, len(User_Neighbor[0]) + len(Item_Neighbor[0])))  # 只包含该区域用户的评分数据
    User_Neighbor = np.array(User_Neighbor)
    Item_Neighbor = np.array(Item_Neighbor)
    Mean_User = np.nan_to_num(np.sum(Rating_Matrix, axis = 1) / np.count_nonzero(Rating_Matrix, axis = 1))  # 根据用户已有的评分求出的用户评价平均分
    Mean_Item = np.nan_to_num(np.sum(Rating_Matrix, axis = 0) / np.count_nonzero(Rating_Matrix, axis = 0))
    Rating_Matrix_User = Rating_Matrix.copy()
    Rating_Matrix_Content = Rating_Matrix.copy().T  # 从item角度的评分矩阵
    """将原评分矩阵中为0的元素替换为均分"""
    for i in range(len(Mean_User)):
        Rating_Matrix_User[i] = np.where(Rating_Matrix_User[i] == 0, Mean_User[i], Rating_Matrix_User[i])

    for i in range(len(Mean_Item)):
        Rating_Matrix_Content[i] = np.where(Rating_Matrix_Content[i] == 0, Mean_Item[i], Rating_Matrix_Content[i])

    for i in range(len(DataSet_Rating)):  # 获取协同评分作为输入
        userid = int(DataSet_Rating[i, 0])
        itemid = int(DataSet_Rating[i, 1])
        tmp1 = Rating_Matrix_User[User_Neighbor[userid - 1] - 1, itemid - 1]  # 邻居用户对该内容的评分
        tmp2 = Rating_Matrix_Content[Item_Neighbor[itemid - 1] - 1, userid - 1]  # 该用户对邻居物品的评分
        collaborative_rating = np.hstack((tmp1, tmp2))
        inputdata = np.vstack((inputdata, collaborative_rating))
    label = DataSet_Rating[:, 2]
    inputdata = np.delete(inputdata, 0, axis = 0)
    return inputdata, label




