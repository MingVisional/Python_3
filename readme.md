# 人工智能初期学习心得
## numpy
用numpy来进行矩阵计算比起python使用循环进行矩阵计算，使用时间更短，代码也更容易编辑。

numpy内置的主要函数及部分运用例举如下：


> np.arange(10)#创建一个0~9的数组
> np.arange(2,10,2)#创建一个从2到10步长为2的数组
>
> np.ones(10)#创建一个全是1的数组
> np.ones( (2,3) )
> np.ones_like(x)#创建一个与x形状相同的全是1的数组
>
> np.zeros(10)
> np.zeros((2,4))
> np.zeros_like(x)
>
> np.empty(10)#创建空数组，里面值可能随机
> np.empty((2,4))
> np.empty_like(x)
>
> np.full(10,666)#创建一个长度10的数组充满666
> np.full((2,4),666)
> np.full_like(x,666)
>
> np.random.randn()#括号内0个参数返回一个数字，n个数字返回n-1维数组
>
> A = np.arange(10).reshape(2,5)#把一个长度为10的数组变为2行5列的数组
> A+1#每个数字+1
> A*3
> np.sin(A)
> np.exp(A)
>
> X[-1,2]#即最后一行，第二个元素
> X[2]#筛选第二行
>
> indexs = np.array([ [0,2],[1.3] ])
> x[indexs]#按照indexs的格式输出对应位置的值
>
> x>5#返回一个与x同形的数组，里面每个数字为对应数字进行运算后的bool值
> x[x>5]#可以筛选出x中所有大于5的数
>
> rand(d0,d1,...,dn) #返回数据在[0,1)之间，具有均匀分布
> randn(d0,d1,...,dn) #返回数据具有标准正态分布
>
> #以下函数都有一个参数axis用于指定计算轴为行或是列
> numpy.sum #所有元素的和
> numpy.prod #所有元素的乘积
> numpy.cumsum #元素的累积和加
> numpy.cumprod #元素的累积乘积
> numpy.min/max #元素的最小/最大值
> numpy.mean #平均值
> numpy.std #标准差
> numpy.var #方差
>
> numpy.arange(12).reshape(3,4)
> numpy.concatenate(array_list，axis)#沿指定axis进行数组合并
> numpy.vstack/numpy.row_stack(array_list) #垂直、按行进行数据合并
> numpy.hstack/numpy.column_stack(array_list) #水平、按列进行数据合并
>


<br>
<br>
<br>

## 神经网络编程
通过模拟神经网络建立模型来实现人工智能。

通过训练神经网络，不断修改输入层、隐藏层、输出层之间的权重来让该神经网络逐渐接近我们所需要的样子。

```python
#创建一个神经网络的对象
class neuralNetwork:
	#进行初始化
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes #此为输入节点的数量
        self.hnodes = hiddennodes #此为隐藏节点的数量
        self.onodes = outputnodes #此为输出阶段的数量

        self.lr = learningrate #此为学习率

        #创建平均值为0，标准方差为1/√传入链接数目 的权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes)) #输入层到隐藏层的权重
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))#隐藏层到输出层的权重
        self.activation_function = lambda x:scipy.special.expit(x) #创建一个函数用于计算1/（1+e^(-x)）
        pass
    #训练神经网络的函数
    def train(self,inputs_list,targets_list):#输入参数为已知前提(inputs_list)和理想目标结果(targets_list)
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs) #计算输入到隐藏层的数据
        hidden_outputs = self.activation_function(hidden_inputs)#计算从隐藏层输出的数据
        final_inputs = np.dot(self.who,hidden_outputs)#计算输入到输出层的数据
        final_outputs = self.activation_function(final_inputs)#计算从输出层输出的数据，即目前模型所得到的的最终数据

        output_errors = targets - final_outputs#计算目标数据和实际最终数据的误差
        hidden_errors = np.dot(self.who.T,output_errors)#计算隐藏层误差

		#根据误差和学习率修改权重，此公式为训练神经网络的关键
        self.who += self.lr * np.dot( output_errors * final_outputs * (1.0 - final_outputs)
                                       , np.transpose(hidden_outputs) )
        self.wih += self.lr * np.dot( hidden_errors*hidden_outputs*(1.0-hidden_outputs)
                                        ,np.transpose(inputs) )

        pass
    #此为训练完神经网络后进行计算的函数
    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin = 2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
```
以上为一个简易神经网络所具备的功能。
以下代码为实际使用该神经网络，让该神经网络可以识别手写数字0~9

每个手写数字的图片为28*28个网格组成。因此输入层的节点即为28*28=784个节点，每个节点代表对应网格的信息。输出节点为10个，分别对应0~9这10个数字。

```python
input_nodes = 784#28*28的网格
hidden_nodes = 200#可改变的隐藏节点数量，使用适合的隐藏节点个数可以提高该人工智能的准确率
output_nodes = 10

learning_rate = 0.1 #设置学习率，适合的学习率也可以提高该人工智能的准确率

epochs = 5#世代数，即对一组已知数据的训练次数，5即为对所有已知的前提循环训练5次，适合的循环次数也可以提高准确率

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate) #实例化一个神经网络

#获取用来训练的数据
training_data_file = open("train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for e in range(epochs):#循环5次
    for record in training_data_list[1:]:
        all_values = record.split(',')

        inputs = (np.asfarray(all_values[1:])/255.0*0.99) +0.01 #把原本0~255的范围缩小为0.01~1的范围
        #建立该手写数字所对应的理想结果
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        #如该数字是2，则targets为[0.01,0.01,0.99,0.01,0.01,......]
        #使用0.01和0.99的原因是为了提高训练的有效性，0在矩阵运算的计算过程中过于绝对，与任何数相乘都为0，可能会对训练造成不利影响

        n.train(inputs,targets)#调用训练函数


#利用训练好的神经网络识别其他手写数字
test_data_file = open("test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

answer_list = []

for record in test_data_list[1:]:
    all_values = record.split(',')

    inputs = (np.asfarray(all_values)/255.0*0.99) +0.01

    answer = n.query(inputs)
    answer_list.append(np.argmax(answer))

print("ImageId,Label")
for i in range(len(answer_list)):
    print(str(i+1) + "," + str(answer_list[i]))
```

最后，可以通过调节学习率、世代数、隐藏层节点个数等来尝试提高该人工智能的正确率。
<br>
<br>
<br>
综上，建立一个所需的人工智能的步骤为：

 1. 创建一个神经网络类
 2. 根据实际情况的需要来决定输入层节点数、输出层节点数
 3. 通过给与大量已知数据来训练神经网络
 4. 使用训练好的神经网络
 5. 通过修改学习率、世代数、隐藏层节点个数、给与更多训练数据，来尝试提高该人工智能的正确率