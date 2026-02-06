# -*- coding: UTF-8 -*-
# Author: WuTong
# Data: 2023.02.18
# 选择用户和内容，生成训练集和测试集，对数据进行预处理
import json
import os

import pandas as pd
from datetime import datetime

from DataProcess import *
import numpy as np

def load_dataset(dataset_name):
    """加载不同数据集的函数"""
    if dataset_name == "filmtrust":
        """读入评分数据 filmtrust"""
        """These files contain 35497 anonymous ratings of approximately 2071 movies
        made by 1508 users. Ratings: [0.5,5]"""
        Original_Ratings = pd.read_table("data/filmtrust/ratings.txt", sep = ' ', header = None, engine = 'python')
        Sequence = list(range(0, len(Original_Ratings)))
        random.shuffle(Sequence)
        Original_Ratings = np.array(Original_Ratings)
        Original_Ratings = Original_Ratings[Sequence, :]
        
    elif dataset_name == "ml_1m":
        """读入评分数据 ml_1m"""
        """These files contain 1,000,209 anonymous ratings of approximately 3,900 movies
        made by 6,040 MovieLens users who joined MovieLens in 2000. Ratings: [1,5]"""
        Original_Ratings = pd.read_table("data/ml-1m/ratings.dat", sep = '::', header = None, engine = 'python')
        Original_Ratings = Original_Ratings.sort_values(by = 3)  # 将评分数据根据时间戳排序
        Original_Ratings = np.array(Original_Ratings)
        
    elif dataset_name == "ml_la":
        """读入评分数据 ml_la"""
        """It contains 100836 ratings and 3683 tag applications across 9724 movies.
        These data were created by 610 users between March 29, 1996 and September 24, 2018.
        This dataset was generated on September 26, 2018. Ratings: [1,5]"""
        Original_Ratings = pd.read_csv("data/ml-la/ratings.csv", header = None)
        Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
        Original_Ratings = np.array(Original_Ratings)  # 原始评分数据
        
    elif dataset_name == "automovie":
        """读入评分数据 Automotive"""
        """It contains 12665 ratings from 5000 users and 6596 items. Ratings: [1,5]"""
        Original_Ratings = pd.read_csv("data/transratings_Automotive.csv", header = None)
        Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
        Original_Ratings = np.array(Original_Ratings)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 处理Movie_ID和User_ID
    Movie_ID = np.sort(np.unique(Original_Ratings[:, 1]).astype(int))   # 实际的内容ID
    Movie_ID_tmp = np.array(list(range(1, len(Movie_ID) + 1, 1)))
    Movie_ID = np.vstack((Movie_ID, Movie_ID_tmp)).T  # 第二列是编号，在实际计算中编号来计算
    User_ID = np.sort(np.unique(Original_Ratings[:, 0]).astype(int))  # 实际的用户ID
    User_ID_tmp = np.array(list(range(1, len(User_ID) + 1, 1)))
    User_ID = np.vstack((User_ID, User_ID_tmp)).T  # 第一列是数据集中的ID
    """将原始数据中的MovieID和UserID进行整理，变成按顺序排列的数字"""
    for i in range(len(User_ID)):
        User_id_tmp = User_ID[i,0]   # 每一个用户在数据集中的ID
        User_index = np.where(Original_Ratings[:,0] == User_id_tmp)[0]  # 找出原始数据中所有该用户的索引
        User_index_rel = np.where(User_ID[:,0] == User_id_tmp)[0]
        Original_Ratings[User_index,0] = User_ID[User_index_rel,1]  # 将重新排序的用户ID赋予该用户

    User_ID[:,0] = User_ID[:,1]

    """同上"""
    for i in range(len(Movie_ID)):
        Movie_ID_tmp = Movie_ID[i,0]
        Movie_index = np.where(Original_Ratings[:,1] == Movie_ID_tmp)[0]
        Movie_index_rel = np.where(Movie_ID[:,0] == Movie_ID_tmp)[0]
        Original_Ratings[Movie_index,1] = Movie_ID[Movie_index_rel,1]

    Movie_ID[:,0] = Movie_ID[:,1]
    
    return Original_Ratings, User_ID, Movie_ID


if __name__ == '__main__':
    start = datetime.now()  # 此时的时间
    
    # 定义要处理的数据集列表
    datasets = ["filmtrust", "ml_1m", "ml_la", "automovie"]
    
    # 循环处理每个数据集
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # 加载数据集
            Original_Ratings, User_ID, Movie_ID = load_dataset(dataset_name)
            
            """基本设置"""
            Num_Cluster = 6 # 群落数量
            k_user = 5  # 邻近用户数
            k_item = 5  # 邻近内容数
            
            # 训练集比例列表
            Train_Test_Rates = [0.25, 0.5, 0.75]
            
            # 循环处理不同的训练集比例，为每个比例生成数据并保存到不同目录
            for Train_Test_Rate in Train_Test_Rates:
                print(f"\n处理数据集 {dataset_name} - 训练集比例: {Train_Test_Rate}")
                
                Num_User = len(np.unique(Original_Ratings[:, 0]))  # 训练集用户数
                Num_Content = len(np.unique(Original_Ratings[:, 1]))  # 训练集内容数
                data_train = Original_Ratings[0:round(Train_Test_Rate * len(Original_Ratings[:, 0])), :]
                data_test = Original_Ratings[round(Train_Test_Rate * len(Original_Ratings[:, 0])):len(Original_Ratings[:, 0]) + 1,:]
                User_Group = Get_User_Group(User_ID,Num_Cluster)
                # Num_Rating = max(Original_Ratings[:,2])/min(Original_Ratings[:,2])

                """获取用于训练的电影相关数据集"""
                DataSetTrain_Rating, Rating_Matrix_Train = DataProcessMovies(data_train,User_ID,Movie_ID) #训练集的评分数据和评分矩阵
                DataSetTest_Rating, Rating_Matrix_Test = DataProcessMovies(data_test, User_ID,Movie_ID) #测试集的评分数据和评分矩阵
                DataSetTrain_Rating = pd.DataFrame(DataSetTrain_Rating)
                DataSetTest_Rating = pd.DataFrame(DataSetTest_Rating)
                Original_Ratings_df = pd.DataFrame(Original_Ratings)
                # DataCategoryOneHot = pd.DataFrame(DataCategoryOneHot)
                User_ID_df = pd.DataFrame(User_ID)
                Movie_ID_df = pd.DataFrame(Movie_ID)

                # 为每个数据集和比例创建对应的目录名
                ratio_str = str(Train_Test_Rate).replace('.', '_')
                data_dir = f"train_data_{dataset_name}_{ratio_str}"
                os.makedirs(data_dir, exist_ok=True)

                """保存原始评分矩阵和评分信息到对应比例的目录"""
                Rating_Matrix_Train.to_csv(f"{data_dir}/Rating_Matrix_Train.csv")
                Rating_Matrix_Test.to_csv(f"{data_dir}/Rating_Matrix_Test.csv")
                DataSetTrain_Rating.to_csv(f"{data_dir}/DataSetTrain_Rating.csv")
                DataSetTest_Rating.to_csv(f"{data_dir}/DataSetTest_Rating.csv")
                Original_Ratings_df.to_csv(f"{data_dir}/Original_Ratings.csv")
                # DataCategoryOneHot.to_csv(f"{data_dir}/DataCategoryOneHot.csv")
                User_ID_df.to_csv(f"{data_dir}/User_ID.csv")
                Movie_ID_df.to_csv(f"{data_dir}/Movie_ID.csv")

                with open(f"{data_dir}/User_Group.txt",'w') as file:
                    for item in User_Group:
                        file.write(str(item) + "\n")
                
                print(f"数据集 {dataset_name} - 训练集比例 {Train_Test_Rate} 的数据已保存到 {data_dir} 目录")
        
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    end = datetime.now()
    print(f"\n所有数据集处理完成！")
    print(f"开始时间: {start}")
    print(f"结束时间: {end}")


    # """读入评分数据 ml_1m"""
    # """These files contain 1,000,209 anonymous ratings of approximately 3,900 movies
    # made by 6,040 MovieLens users who joined MovieLens in 2000. Ratings: [1,5]"""
    # Original_Ratings = pd.read_table("DataSet/ratings.dat", sep = '::', header = None, engine = 'python')
    # Original_Ratings = Original_Ratings.sort_values(by = 3)  # 将评分数据根据时间戳排序
    # Original_Ratings = np.array(Original_Ratings)
    #
    # Movie_ID = np.sort(np.unique(Original_Ratings[:, 1]).astype(int))   # 实际的内容ID
    # Movie_ID_tmp = np.array(list(range(1, len(Movie_ID) + 1, 1)))
    # Movie_ID = np.vstack((Movie_ID, Movie_ID_tmp)).T  # 第二列是编号，在实际计算中编号来计算
    # User_ID = np.sort(np.unique(Original_Ratings[:, 0]).astype(int))  # 实际的用户ID
    # User_ID_tmp = np.array(list(range(1, len(User_ID) + 1, 1)))
    # User_ID = np.vstack((User_ID, User_ID_tmp)).T  # 第一列是数据集中的ID
    # """将原始数据中的MovieID和UserID进行整理，变成按顺序排列的数字"""
    # for i in range(len(User_ID)):
    #     User_id_tmp = User_ID[i,0]   # 每一个用户在数据集中的ID
    #     User_index = np.where(Original_Ratings[:,0] == User_id_tmp)[0]  # 找出原始数据中所有该用户的索引
    #     User_index_rel = np.where(User_ID[:,0] == User_id_tmp)[0]
    #     Original_Ratings[User_index,0] = User_ID[User_index_rel,1]  # 将重新排序的用户ID赋予该用户
    #
    # User_ID[:,0] = User_ID[:,1]
    #
    # """同上"""
    # for i in range(len(Movie_ID)):
    #     Movie_ID_tmp = Movie_ID[i,0]
    #     Movie_index = np.where(Original_Ratings[:,1] == Movie_ID_tmp)[0]
    #     Movie_index_rel = np.where(Movie_ID[:,0] == Movie_ID_tmp)[0]
    #     Original_Ratings[Movie_index,1] = Movie_ID[Movie_index_rel,1]
    #
    # Movie_ID[:,0] = Movie_ID[:,1]

    # """读入评分数据 MT. Ratings: [1,10]"""
    # Original_Ratings = pd.read_table("data/MT/ratings.dat", sep='::', header=None, engine='python')
    # Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
    # Original_Ratings = np.array(Original_Ratings)

    # """读入评分数据 ml_100k"""
    # """This data set consists of:
	# * 100,000 ratings (1-5) from 943 users on 1682 movies.
	# * Each user has rated at least 20 movies. Ratings: [1,5]"""
    # Original_Ratings = pd.read_table("data/ml-100k/u.data", sep='\t', header=None, engine='python')
    # Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
    # Original_Ratings = np.array(Original_Ratings)

    # """读入评分数据 ml_la"""
    # """It contains 100836 ratings and 3683 tag applications across 9724 movies.
    # These data were created by 610 users between March 29, 1996 and September 24, 2018.
    # This dataset was generated on September 26, 2018. Ratings: [1,5]"""
    # Original_Ratings = pd.read_csv("data/ml-la/ratings.csv", header = None)
    # Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
    # Original_Ratings = np.array(Original_Ratings)  # 原始评分数据
    # Movie_ID = np.sort(np.unique(Original_Ratings[:, 1]).astype(int))   # 实际的内容ID
    # Movie_ID_tmp = np.array(list(range(1, len(Movie_ID) + 1, 1)))
    # Movie_ID = np.vstack((Movie_ID, Movie_ID_tmp)).T  # 第二列是编号，在实际计算中编号来计算
    # User_ID = np.sort(np.unique(Original_Ratings[:, 0]).astype(int))  # 实际的用户ID
    # User_ID_tmp = np.array(list(range(1, len(User_ID) + 1, 1)))
    # User_ID = np.vstack((User_ID, User_ID_tmp)).T  # 第一列是数据集中的ID
    # """将原始数据中的MovieID和UserID进行整理，变成按顺序排列的数字"""
    # for i in range(len(User_ID)):
    #     User_id_tmp = User_ID[i,0]   # 每一个用户在数据集中的ID
    #     User_index = np.where(Original_Ratings[:,0] == User_id_tmp)[0]  # 找出原始数据中所有该用户的索引
    #     User_index_rel = np.where(User_ID[:,0] == User_id_tmp)[0]
    #     Original_Ratings[User_index,0] = User_ID[User_index_rel,1]  # 将重新排序的用户ID赋予该用户
    #
    # User_ID[:,0] = User_ID[:,1]
    #
    # """同上"""
    # for i in range(len(Movie_ID)):
    #     Movie_ID_tmp = Movie_ID[i,0]
    #     Movie_index = np.where(Original_Ratings[:,1] == Movie_ID_tmp)[0]
    #     Movie_index_rel = np.where(Movie_ID[:,0] == Movie_ID_tmp)[0]
    #     Original_Ratings[Movie_index,1] = Movie_ID[Movie_index_rel,1]
    #
    # Movie_ID[:,0] = Movie_ID[:,1]


    # """读入评分数据 A-Baby"""
    # """It contains 17532 ratings from 6000 users and 5428 itmes. Ratings: [1,5]"""
    # Original_Ratings = pd.read_csv("data/transratings_Baby.csv", header = None)
    # Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
    # Original_Ratings = np.array(Original_Ratings)

    # """读入评分数据 A-Digital-Music"""
    # """It contains 11677 ratings from 1000 users and 4796 itmes. Ratings: [1,5]"""
    # Original_Ratings = pd.read_csv("data/transratings_Digital_Music.csv", header = None)
    # Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
    # Original_Ratings.to_csv("train_data/Original_Ratings.csv")
    # Original_Ratings = np.array(Original_Ratings)
    #
    # Movie_ID = np.sort(np.unique(Original_Ratings[:, 1]).astype(int))  # 实际的内容ID
    # Movie_ID_tmp = np.array(list(range(1, len(Movie_ID) + 1, 1)))
    # Movie_ID = np.vstack((Movie_ID, Movie_ID_tmp)).T  # 第二列是编号，在实际计算中编号来计算
    # User_ID = np.sort(np.unique(Original_Ratings[:, 0]).astype(int))  # 实际的用户ID
    # User_ID_tmp = np.array(list(range(1, len(User_ID) + 1, 1)))
    # User_ID = np.vstack((User_ID, User_ID_tmp)).T  # 第一列是数据集中的ID
    # """将原始数据中的MovieID和UserID进行整理，变成按顺序排列的数字"""
    # for i in range(len(User_ID)):
    #     User_id_tmp = User_ID[i, 0]  # 每一个用户在数据集中的ID
    #     User_index = np.where(Original_Ratings[:, 0] == User_id_tmp)[0]  # 找出原始数据中所有该用户的索引
    #     User_index_rel = np.where(User_ID[:, 0] == User_id_tmp)[0]
    #     Original_Ratings[User_index, 0] = User_ID[User_index_rel, 1]  # 将重新排序的用户ID赋予该用户
    #
    # User_ID[:, 0] = User_ID[:, 1]
    #
    # """同上"""
    # for i in range(len(Movie_ID)):
    #     Movie_ID_tmp = Movie_ID[i, 0]
    #     Movie_index = np.where(Original_Ratings[:, 1] == Movie_ID_tmp)[0]
    #     Movie_index_rel = np.where(Movie_ID[:, 0] == Movie_ID_tmp)[0]
    #     Original_Ratings[Movie_index, 1] = Movie_ID[Movie_index_rel, 1]
    #
    # Movie_ID[:, 0] = Movie_ID[:, 1]


    # """读入评分数据 A-Automotive"""
    # """It contains 12665 ratings from 5000 users and 6596 itmes. Ratings: [1,5]"""
    # Original_Ratings = pd.read_csv("data/transratings_Automotive.csv", header = None)
    # Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
    # Original_Ratings = np.array(Original_Ratings)
    #
    # Movie_ID = np.sort(np.unique(Original_Ratings[:, 1]).astype(int))  # 实际的内容ID
    # Movie_ID_tmp = np.array(list(range(1, len(Movie_ID) + 1, 1)))
    # Movie_ID = np.vstack((Movie_ID, Movie_ID_tmp)).T  # 第二列是编号，在实际计算中编号来计算
    # User_ID = np.sort(np.unique(Original_Ratings[:, 0]).astype(int))  # 实际的用户ID
    # User_ID_tmp = np.array(list(range(1, len(User_ID) + 1, 1)))
    # User_ID = np.vstack((User_ID, User_ID_tmp)).T  # 第一列是数据集中的ID
    # """将原始数据中的MovieID和UserID进行整理，变成按顺序排列的数字"""
    # for i in range(len(User_ID)):
    #     User_id_tmp = User_ID[i, 0]  # 每一个用户在数据集中的ID
    #     User_index = np.where(Original_Ratings[:, 0] == User_id_tmp)[0]  # 找出原始数据中所有该用户的索引
    #     User_index_rel = np.where(User_ID[:, 0] == User_id_tmp)[0]
    #     Original_Ratings[User_index, 0] = User_ID[User_index_rel, 1]  # 将重新排序的用户ID赋予该用户
    #
    # User_ID[:, 0] = User_ID[:, 1]
    #
    # """同上"""
    # for i in range(len(Movie_ID)):
    #     Movie_ID_tmp = Movie_ID[i, 0]
    #     Movie_index = np.where(Original_Ratings[:, 1] == Movie_ID_tmp)[0]
    #     Movie_index_rel = np.where(Movie_ID[:, 0] == Movie_ID_tmp)[0]
    #     Original_Ratings[Movie_index, 1] = Movie_ID[Movie_index_rel, 1]
    #
    # Movie_ID[:, 0] = Movie_ID[:, 1]

    # """读入评分数据 ml_10M"""
    # """This data set contains 10000054 ratings and 95580 tags applied to 10681
    # movies by 71567 users of the online movie recommender service MovieLens. Ratings: [1,5]"""
    # Original_Ratings = pd.read_table("data/ml-10M/ratings.dat", sep='::', header=None, engine='python')
    # Original_Ratings = Original_Ratings.sort_values(by=3)  # 将评分数据根据时间戳排序
    # Original_Ratings = np.array(Original_Ratings)
    #
    # Movie_ID = np.sort(np.unique(Original_Ratings[:, 1]).astype(int))  # 实际的内容ID
    # Movie_ID_tmp = np.array(list(range(1, len(Movie_ID) + 1, 1)))
    # Movie_ID = np.vstack((Movie_ID, Movie_ID_tmp)).T  # 第二列是编号，在实际计算中编号来计算
    # User_ID = np.sort(np.unique(Original_Ratings[:, 0]).astype(int))  # 实际的用户ID
    # User_ID_tmp = np.array(list(range(1, len(User_ID) + 1, 1)))
    # User_ID = np.vstack((User_ID, User_ID_tmp)).T  # 第一列是数据集中的ID
    # """将原始数据中的MovieID和UserID进行整理，变成按顺序排列的数字"""
    # for i in range(len(User_ID)):
    #     User_id_tmp = User_ID[i, 0]  # 每一个用户在数据集中的ID
    #     User_index = np.where(Original_Ratings[:, 0] == User_id_tmp)[0]  # 找出原始数据中所有该用户的索引
    #     User_index_rel = np.where(User_ID[:, 0] == User_id_tmp)[0]
    #     Original_Ratings[User_index, 0] = User_ID[User_index_rel, 1]  # 将重新排序的用户ID赋予该用户
    #
    # User_ID[:, 0] = User_ID[:, 1]

    # """同上"""
    # for i in range(len(Movie_ID)):
    #     Movie_ID_tmp = Movie_ID[i, 0]
    #     Movie_index = np.where(Original_Ratings[:, 1] == Movie_ID_tmp)[0]
    #     Movie_index_rel = np.where(Movie_ID[:, 0] == Movie_ID_tmp)[0]
    #     Original_Ratings[Movie_index, 1] = Movie_ID[Movie_index_rel, 1]
    #
    # Movie_ID[:, 0] = Movie_ID[:, 1]









    # """利用协同过滤得到输入和标签"""
    # traindata, trainlabel = CF(k_item, k_user,Num_User, Num_Content, Rating_Matrix_Train, User_ID, Movie_ID, MeanUser_Train, MeanContent_Train)
    #
    # """testdata: 测试阶段的输入; testlabel: 测试阶段的标签"""
    # testdata, testlabel = CF(k_item, k_user, Num_User, Num_Content, Rating_Matrix_Test, User_ID, Movie_ID, MeanUser_Train, MeanContent_Train)
    #
    # traindata = pd.DataFrame(traindata)
    # trainlabel = pd.DataFrame(trainlabel)
    # Rating_Matrix_Train = pd.DataFrame(Rating_Matrix_Train)  #修改
    # testdata = pd.DataFrame(testdata)
    # testlabel = pd.DataFrame(testlabel)
    # Rating_Matrix_Test = pd.DataFrame(Rating_Matrix_Test)  #修改
    #
    # """保存处理过后的数据"""
    # traindata.to_csv("train_data/traindata.csv")
    # trainlabel.to_csv("train_data/trainlabel.csv")
    # Rating_Matrix_Train.to_csv("train_data/Rating_Matrix_train.csv")
    # testdata.to_csv("train_data/testdata.csv")
    # testlabel.to_csv("train_data/testlabel.csv")
    # Rating_Matrix_Test.to_csv("train_data/Rating_Matrix_Test.csv")
    #
    #
    # """保存跨域推荐的数据集"""
    # traindata.to_csv("Cross_Domain/Automotive_traindata.csv")
    # trainlabel.to_csv("Cross_Domain/Automotive_trainlabel.csv")
    # testdata.to_csv("Cross_Domain/Automotive_testdata.csv")
    # testlabel.to_csv("Cross_Domain/Automotive_testlabel.csv")

    end = datetime.now()
    print(start, end)




