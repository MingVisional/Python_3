import numpy as np
import scipy.special#函数expit()为S函数
import matplotlib.pyplot

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        #创建平均值为0，标准方差为1/√传入链接数目 的权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes)) #输入层到隐藏层的权重
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))#隐藏层到输出层的权重
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)

        self.who += self.lr * np.dot( output_errors * final_outputs * (1.0 - final_outputs)
                                       , np.transpose(hidden_outputs) )
        self.wih += self.lr * np.dot( hidden_errors*hidden_outputs*(1.0-hidden_outputs)
                                        ,np.transpose(inputs) )

        pass
    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin = 2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 784#28*28的网格
hidden_nodes = 200#可改变的隐藏节点数量
output_nodes = 10

learning_rate = 0.1

epochs = 5#世代数

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#print(n.query([1.0,0.5,-1.5]))

#获取数据
training_data_file = open("train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#all_values = training_data_list[1].split(",")
#image_array = np.asfarray(all_values[1:]).reshape((28,28))#asfarray函数将文本字符串转为实数然后创建一个数组
#matplotlib.pyplot.imshow(image_array,cmap = 'Greys',interpolation = 'None')
#matplotlib.pyplot.show()

# scaled_input = (np.asfarray(all_values[1:])/255.0*0.99) +0.01 #将0~255的范围转化为0.01~1.00

for e in range(epochs):
    for record in training_data_list[1:]:
        all_values = record.split(',')

        inputs = (np.asfarray(all_values[1:])/255.0*0.99) +0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        n.train(inputs,targets)


#测试
test_data_file = open("test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

answer_list = []

for record in test_data_list[1:]:
    all_values = record.split(',')

    inputs = (np.asfarray(all_values)/255.0*0.99) +0.01

    answer = n.query(inputs)
    # maxNum = 0
    # for i in range(0,len(answer)):
    #     if(answer[i] > maxNum):
    #         maxNum = answer[i]
    #         flag = i
    # answer_list.append(flag)
    answer_list.append(np.argmax(answer))


print("ImageId,Label")
for i in range(len(answer_list)):
    print(str(i+1) + "," + str(answer_list[i]))


# all_values = test_data_list[1].split(",")
# image_array = np.asfarray(all_values).reshape((28,28))#asfarray函数将文本字符串转为实数然后创建一个数组
# matplotlib.pyplot.imshow(image_array,cmap = 'Greys',interpolation = 'None')
# matplotlib.pyplot.show()
#
# inputs = (np.asfarray(all_values)/255.0*0.99) +0.01
# print(n.query(inputs))
