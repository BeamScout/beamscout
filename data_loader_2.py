import sys
import numpy as np
import os
import operator
import random

class dataset(object):
    def __init__(self, dir):
        self.data_dir=dir
        self.files = self.get_files()
        self.seq = [[0,0],[0,1],[0,1],[0,2],[0,2],[0,3],[0,3],[0,4],[0,4],[0,5],[0,5],[0,6],[0,6],[0,7],[0,7],
                    [1,1],[1,2],[1,2],[1,3],[1,3],[1,4],[1,4],[1,5],[1,5],[1,6],[1,6],[1,7],[1,7],
                    [2,2],[2,3],[2,3],[2,4],[2,4],[2,5],[2,5],[2,6],[2,6],[2,7],[2,7],
                    [3,3],[3,4],[3,4],[3,5],[3,5],[3,6],[3,6],[3,7],[3,7],
                    [4,4],[4,5],[4,5],[4,6],[4,6],[4,7],[4,7],
                    [5,5],[5,6],[5,6],[5,7],[5,7],
                    [6,6],[6,7],[6,7],
                    [7,7]
                    ]#将8*8复数矩阵压缩成64维向量的映射关系
        self.data_s_list = self.get_data_list(self.files)
        self.R_in_oracle = self.get_in_array(self.files) #纯干扰信号矩阵的中间矩阵
        self.train_data_pair = self.get_train_data(self.data_s_list) #混杂信号
    def get_files(self):
        files = []
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            files.append(file_path)
        return files

    def get_data_list(self, files):
        data_list = []  #8*8*512
        for file in files:
            f = open(file,'r')
            lines = f.readlines()
            data = []
            for line in lines:
                data += [float(i) for i in line.split(" ") if i.strip()]
                # if(len(data)!=16):
                #     print(len(data))
                #print(len(data))
            for i in range(0,len(data)):
                if(int(i/128)%1024>=512):
                    data_list.append(data[i])
            #print(len(data_list))
        return data_list
    def get_train_data(self, data_list):
        data_all = []
        for i in range(0,int(len(data_list)/128)):
            data_train = i%512
            data_label = []
            test = []
            data_label_ = data_list[i*128:i*128+128]
            #print(len(data_label_))
            for k in range(0,64):
                col = int(k/8)
                row = int(k)%8
                if(col == row):
                    data_label.append(data_label_[k*2])

                if(col < row):
                    data_label.append(data_label_[k*2])
                    data_label.append(data_label_[k*2+1])

            #print(test)
            data_pair = [data_train,data_label]
            data_all.append(data_pair)
        return data_all

    def get_in_array(self,files):
        data_list=[]
        data_complex = []
        data = []
        for file in files:
            f = open(file,'r')
            lines = f.readlines()
            data += [float(i) for line in lines for i in line.split(" ") if i.strip()]
        for i in range(0,len(data)):
            if(int(i/128)%1024==256):
                data_list.append(data[i])
        data_real = data_list[::2]
        data_imag = data_list[1::2]
        for i in range(0,len(data_real)):
            data_complex.append(complex(data_real[i],data_imag[i]))
        return data_complex

    def vec_to_array(self,data_vec):
        data_seq = self.seq
        array = [[complex(0,0)]*8 for i in range(8)]
        for i in range(0,64):
            col = self.seq[i][0]
            row = self.seq[i][1]
            if(col == row):
                array[col][row]=complex(data_vec[i],0)
            else:
                if(operator.eq(data_seq[i],data_seq[i+1])):
                    if(col<row):
                        array[col][row]=complex(data_vec[i],data_vec[i+1])
                        array[row][col]=complex(data_vec[i],-data_vec[i+1])
                    else:
                        array[col][row]=complex(data_vec[i],-data_vec[i+1])
                        array[row][col]=complex(data_vec[i],data_vec[i+1])
                else:
                    continue
        return array
if __name__ == '__main__':
    data = dataset('.\\cov1_test')





    #print(files)