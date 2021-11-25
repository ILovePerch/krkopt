
from libsvm.svm import svm_parameter, svm_problem
from libsvm.svmutil import *
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_save_model
from libsvm.svmutil import svm_load_model
from libsvm.svmutil import svm_predict
import numpy as np

##Read file and save data as 2d matrix

def read_file(file):
     num_list =[]
     with open(file, "r",encoding='UTF-8-sig')as fi:
         for l in fi:
             l = l.split(",")
             list_a=[]
             for j in range(3):
                 list_a.append(ord(l[j*2])-ord("a"))
                 list_a.append(ord(l[j*2+1])-ord("0"))
             if(l[6][0]=="d"):
                 list_a.append("0")
             else:
                 list_a.append(1)
             num_list.append(list_a)
             num_mat = np.array(num_list,dtype="float")
     return(num_mat)
## set training data and test data，5000 training data， rest test
def data_deal(mat,len_train,len1,len_test,len2):
     np.random.shuffle(mat)
     x_part1 = mat[0:len_train,0:len1]
     x_part2 = mat[len_train:,0:len1]
     y_part1 = mat[0:len_train,len1]
     y_part2 = mat[len_train:,len1]
     #normalization
     avgX = np.mean(x_part1)
     stdX = np.std(x_part1)
     for data in x_part1:
         for j in range(len(data)):
             data[j] = (data[j] - avgX) / stdX
     for data in x_part2:
         for j in range(len(data)):
             data[j] = (data[j] - avgX) / stdX
     return x_part1,y_part1,x_part2,y_part2
##SVM Gaussian kernel
def TrainModel(CScale,gammaScale,prob):
    maxACC=0
    maxACC_C=0
    maxACC_gamma=0
    for C in CScale:
        C_=pow(2, C)
        for gamma in gammaScale:
             gamma_ = pow(2, gamma)
             param = svm_parameter('-t 2 -c ' + str(C_) + ' -g ' + str(gamma_) + ' -v 5 -q')
             ACC = svm_train(prob, param) 
             if (ACC > maxACC):
                 maxACC = ACC
                 maxACC_C = C
                 maxACC_gamma = gamma
    return maxACC,maxACC_C,maxACC_gamma
def getNewList(L,U,step):
     l = []
     while(L < U):
         l.append(L)
         L += step
     return l
def TrainModelSVM(data,label,iter,model_file):
     X = data.tolist()
     Y = label.tolist()
     CScale = [-5, -3, -1, 1, 3, 5,7,9,11,13,15]
     gammaScale = [-15,-13,-11,-9,-7,-5,-3,-1,1,3]
     cnt = iter
     step = 2
     maxACC = 0
     bestACC_C = 0
     bestACC_gamma = 0
     prob = svm_problem(Y, X)
     while(cnt):
         maxACC_train,maxACC_C_train,maxACC_gamma_train = TrainModel(CScale,gammaScale,prob)
         if(maxACC_train > maxACC):
             maxACC = maxACC_train
             bestACC_C = maxACC_C_train
             bestACC_gamma = maxACC_gamma_train
         new_step = step*2/10
         CScale = getNewList(maxACC_C_train - step,maxACC_C_train + step + new_step,new_step)
         gammaScale = getNewList(maxACC_gamma_train - step,maxACC_gamma_train + step + new_step,new_step)
         cnt -= 1
     C = pow(2,bestACC_C)
     gamma = pow(2,bestACC_gamma)
     param = svm_parameter('-t 2 -c ' + str(C) + ' -g ' + str(gamma))
     model = svm_train(prob, param)
     svm_save_model(model_file, model)
     return model
def main():
     file = r"C:\\Users\haoni\Downloads\krkopt.data"
     model_file = r"C:\\Users\haoni\Downloads\model_file.txt"
     mat=read_file(file)
     len_train= 5000
     len_test=len(mat) - 5000
     len1=6
     len2=len(mat[0] - len1)
     iter=2
     x_train,y_train,x_test,y_test = data_deal(mat,len_train,len1,len_test,len2)
     if (input("need?") == 'y'):
         model = TrainModelSVM(x_train,y_train,iter,mat) 
     else:
         model = svm_load_model(mat)  
     X = x_test.tolist()
     Y = y_test.tolist()
     p_labs,p_acc,p_vals = svm_predict(Y,X,model)
if __name__ == "__main__":
    main()