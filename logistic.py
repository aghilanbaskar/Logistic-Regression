import pandas as pd
import numpy as np 

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


df = pd.read_csv('Iris.csv') ## Load data
df = df.drop(['Id'],axis=1)
data = df.values.tolist()
data=np.array(data)
np.random.shuffle(data)
train, test = data[:80,:], data[80:,:]

x_train=train[:,:4]
y_train=train[:,4]

x_test=test[:,:4]
y_test=test[:,4]

row,col=np.shape(x_train)
x_1=x_train[:,0]
x_2=x_train[:,1]
x_3=x_train[:,2]
x_4=x_train[:,3]

#reshape dataset
x_1 = x_1.reshape(row,1)
x_2 = x_2.reshape(row,1)
x_3 = x_3.reshape(row,1)
x_4 = x_4.reshape(row,1)
y_train = y_train.reshape(row,1)

l_r=0.0001 #learning rate

theta_0 = np.zeros((row,1))
theta_1 = np.zeros((row,1))
theta_2 = np.zeros((row,1))
theta_3 = np.zeros((row,1))
theta_4 = np.zeros((row,1))

#finding theta
cost_func=[]
count=0
while count<10000:  #count<1000 then accuracy=0.5
    y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * x_3 + theta_4 * x_4
    y=sigmoid(y)
    cost = (- np.dot(np.transpose(y_train),np.log(y)) - np.dot(np.transpose(1-y_train),np.log(1-y)))/row
    theta_0_grad = np.dot(np.ones((1,row)),y-y_train)/row
    theta_1_grad = np.dot(np.transpose(x_1),y-y_train)/row
    theta_2_grad = np.dot(np.transpose(x_2),y-y_train)/row
    theta_3_grad = np.dot(np.transpose(x_3),y-y_train)/row
    theta_4_grad = np.dot(np.transpose(x_4),y-y_train)/row

    theta_0 = theta_0 - l_r * theta_0_grad
    theta_1 = theta_1 - l_r * theta_1_grad
    theta_2 = theta_2 - l_r * theta_2_grad
    theta_3 = theta_3 - l_r * theta_3_grad
    theta_4 = theta_4 - l_r * theta_4_grad

    cost_func.append(cost)
    count += 1

theta_0=np.sum(theta_0)/row
theta_1=np.sum(theta_1)/row
theta_2=np.sum(theta_2)/row
theta_3=np.sum(theta_3)/row
theta_4=np.sum(theta_4)/row
print('t_0',theta_0,'\nt_1',theta_1,'\nt_2',theta_2,'\nt_3',theta_3,'\nt_4',theta_4)


#testing the test dataset
test_x_1 = x_test[:,0]
test_x_2 = x_test[:,1]
test_x_3 = x_test[:,2]
test_x_4 = x_test[:,3]

row,col=np.shape(x_test)
y_predict=[]
for i in range(row):
    y_pred=theta_0+theta_1*test_x_1[i]+theta_2*test_x_2[i]+theta_3*test_x_3[i]+theta_4*test_x_4[i]
    y_pred=sigmoid(y_pred)
    if y_pred >0.5:
        print('y_pred:','1.0 ---',y_test[i],'y_test')
        y_predict.append(1)
    else:
        print('y_pred:','0.0 ---',y_test[i],'y_test')
        y_predict.append(0)

acc=0
for i in range(row):
    if y_predict[i] == y_test[i]:
        acc+=1
print('accuracy:',acc/row)
