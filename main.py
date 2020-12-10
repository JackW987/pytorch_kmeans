import torch 
import matplotlib.pyplot as plt
import numpy as np
import math
#data
clu_1 = torch.randn(200,2)
clu_2 = torch.randn(200,2)+3
clu_3 = torch.randn(200,2)+6
x = torch.cat((clu_1[:,0],clu_2[:,0],clu_3[:,0]),0)
y = torch.cat((clu_1[:,1],clu_2[:,1],clu_3[:,1]),0)
z = torch.Tensor(600).zero_() 
point = torch.stack((x,y,z),1)
point_num= list(point.size())[0]
plt.plot(x,y,'.')
plt.show()
#distance
def distance(cur_point,cls_1,cls_2,cls_3):
    x = cur_point[0]
    y = cur_point[1]
    cls_1_x = cls_1[0]
    cls_1_y = cls_1[1]
    cls_2_x = cls_1[0]
    cls_2_y = cls_1[1]
    cls_3_x = cls_1[0]
    cls_3_y = cls_1[1]
    dis_1 = math.sqrt(pow(x-cls_1_x,2)+pow(y-cls_1[1],2))
    dis_2 = math.sqrt(pow(x-cls_2_x,2)+pow(y-cls_2[1],2))
    dis_3 = math.sqrt(pow(x-cls_3_x,2)+pow(y-cls_3[1],2))
    p = [dis_1,dis_2,dis_3]
    w = p.index(min(p))
    return w
#find_point
def find_point(point,new_1,new_2,new_3,count):
    if point[2] == 0:
        new_1[0] += point[0]
        new_1[1] += point[1]
        count[0] += 1
    elif point[2] == 1:
        new_2[0] += point[0]
        new_2[1] += point[1]
        count[1] += 1
    elif point[2] == 2:
        new_3[0] += point[0]
        new_3[1] += point[1]
        count[2] += 1
#kmeans
def show(point_num):
    result_1 = torch.zeros(1,3)
    result_2 = torch.zeros(1,3)
    result_3 = torch.zeros(1,3)
    for i in range(point_num):
        if point[i,2] == 0:
            result_1 = torch.cat((result_1,point[i,:].unsqueeze(0)),0)
        if point[i,2] == 1:
            result_2 = torch.cat((result_2,point[i,:].unsqueeze(0)),0)
        if point[i,2] == 2:
            result_3 = torch.cat((result_3,point[i,:].unsqueeze(0)),0)
    # tensor selected to remove the first row [0,0,0]
    mask = torch.linspace(1,result_1.size()[0]-1,result_1.size()[0]-1,dtype=int)
    result_1 = torch.index_select(result_1,0,mask)
    mask = torch.linspace(1,result_2.size()[0]-1,result_2.size()[0]-1,dtype=int)
    result_2 = torch.index_select(result_2,0,mask)
    mask = torch.linspace(1,result_3.size()[0]-1,result_3.size()[0]-1,dtype=int)
    result_3 = torch.index_select(result_3,0,mask)
    #show the result
    plt.plot(result_1[:,0],result_1[:,1],'.',color = 'r')
    plt.plot(result_2[:,0],result_2[:,1],'.',color = 'g')
    plt.plot(result_3[:,0],result_3[:,1],'.',color = 'b')
    plt.show()

def kmeans(k,data,ire):
    #class num
    cls_num = torch.linspace(1,k,k)
    #point ire times
    point_num= list(data.size())[0]
    #first random center
    index_1 = torch.rand(1,k)*point_num
    index_1 = torch.floor(index_1[:,:])
    cls_1 = data[int(index_1[0,0]),:]
    cls_2 = data[int(index_1[0,1]),:]
    cls_3 = data[int(index_1[0,2]),:]
    #ire times
    for i in range(ire):
        #one ire in every point
        new_1 = torch.Tensor([0,0,0])
        new_2 = torch.Tensor([0,0,0])
        new_3 = torch.Tensor([0,0,0])
        count = torch.Tensor([0,0,0])
        for j in range(point_num):
            cur_point = data[j,:]
            w = distance(cur_point,cls_1,cls_2,cls_3)
            data[j,2] = w
            find_point(data[j,:],new_1,new_2,new_3,count)
        cls_1 = new_1/count[0]
        cls_2 = new_2/count[1]
        cls_3 = new_3/count[2]
        show(point_num)
#use kmeans
kmeans(3,point,10)





        