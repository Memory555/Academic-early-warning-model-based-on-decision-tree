# 1.数据准备
# 1.1 引入头文件
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus
from six import StringIO
from sklearn.metrics import accuracy_score
# from sklearn.tree import accuracy_score
from sklearn.model_selection import  GridSearchCV

# 1.2 把student_1.csv数据拖入代码的同一文件夹下，同时读取文件中的数据
stu_grade = pd.read_csv('student-mat.csv')
print(stu_grade.head())

# 1.3 特征选取
new_data = stu_grade.iloc[:,:]
print(new_data.head())

# 2.数据处理
# 2.1 对G1、G2、G3处理
# 对于离散值进行连续处理，同时设置lambda函数计算G1、G2、G3。
def choice_2(x):
    x = int(x)
    if x < 5:
        return 'bad'
    elif x >= 5 and x < 10:
        return 'medium'
    elif x >= 10 and x <15:
        return 'good'
    else:
        return 'excellent'

stu_data = new_data.copy()
stu_data['G1'] = pd.Series(map(lambda x:choice_2(x), stu_data['G1']))
stu_data['G2'] = pd.Series(map(lambda x:choice_2(x), stu_data['G2']))
stu_data['G3'] = pd.Series(map(lambda x:choice_2(x), stu_data['G3']))
print(stu_data.head())

# 2.3 由于数据集中每个参数差异比较大，所以这里把特征参数统一改为数字形式
def replace_feature(data): # 把数据处理，字符串改成数字形式
    for each in data.columns: # 遍历data中 的每个fea ture
        feature_list = data[each] # 取出每个feature的数据
        # print(feature_list) #测试
        unique_value = set(feature_list)
        # 剔除每个feature中重复的元素，按收参数为list
        # set输出值的顺序是随机，可能会产生
        # print(unique_value) # 测试，输出结果为不重复的评判标准值
        i= 0
        for fea_value in unique_value: # 遍历单个feature中的每个元素
            data[each] = data[each].replace(fea_value, i)
            # 用数字重置之前每个feature中评判标准的字符串(字符串数值离散化)
            #例如schoo1中的“GP” 评判标准改为0
            # school中的“HS” 评判标准改为1
            i += 1
    return data
stu_data = replace_feature(stu_data)
print(stu_data.head())


# 2.4 对于当前处理过的数据集，划分训练集和测试集，并设置好随机种子等其他参数
X_train, X_test, Y_train, Y_test = train_test_split(stu_data.iloc[:, :-1],
stu_data["G3"], test_size=0.3, random_state=5)
# G3为最终成绩
# stu data. iloc[:, :-1]:除了最后3的元素，其他的作为训练集和测试集
# 划分洲练集和测试集，测试葉占30%
# stu_data[ "G3"]为要划分的样本结果
# test_ size = 0.3为样本占比，如果是整数就是样本的数量
# random_ state=5为随 机数的种子
print(X_test.head())

# 3．训练得到的模型
# 3.1 决策树
# 3.1.1 开始对训练集中的数据进行训练
# 使用sklearn进行沈策树训练
# criterion:选择节点划分质量的废量标准，獻认使用gini,此处设置为entropy
# random state是为保证程序每次运行都分剖-样的训练集和测试集。
# random state就认为None时，则每次程序运行不能保i证分割-一样的训练集和测试集

dt_model = DecisionTreeClassifier(criterion = "entropy", random_state = 666)
dt_model.fit(X_train,Y_train) # 训练模型

# 训练完的模型用来设置图像参数进行可视化展现。
dot_data = tree.export_graphviz(dt_model, out_file = None, # 设置图像参数
                                feature_names = X_train.columns.values,
                                class_names = ['0', ' 1', '2', '3'],
                                filled = True,rounded = True,
                                special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data)

# 可视化决策树
Image(graph.create_png())

# 将图像保存
graph.write_png("dtr.png")

# 3.1.2 利用已经训练好的模型来预测G3的值
# 对训练好的模型进行打分。

# Y_pred = dt_model.predict(X_test) # 模型预测
# print(Y_pred)

# 输出训练集的分数。
Y_pred = dt_model.predict(X_train) # 模型预测
print(Y_pred)

print("训练集",accuracy_score(Y_train, Y_pred))

# 输出测试集的分数。
Y_pred = dt_model.predict(X_test) # 模型预测
print(Y_pred)

print("测试集",accuracy_score(Y_test, Y_pred))


entropy_thresholds = np.linspace (0, 1, 50)
gini_thresholds = np.linspace (0,0.5, 50)

param_grid = [{'criterion':['entropy'],
            'min_impurity_decrease': entropy_thresholds},
              {'criterion':['gini'],
            'min_impurity_decrease':gini_thresholds},
              {'max_depth':range(2, 10)},
            {'min_samples_split':range(2, 30, 2)}]
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, return_train_score = True)
clf.fit(X_train, Y_train)
print("训练集Best param: {0}\nBest score:{1}".format(clf.best_params_,clf.best_score_))


# 3.1.3 对模型中的参数进行优化，输出优化后最好的分数
clf = DecisionTreeClassifier(criterion= 'entropy', min_impurity_decrease= 0.04081632653061224)
clf.fit(X_train, Y_train)

# 3.1.4 优化后的模型来绘制决策树
dot_data = tree.export_graphviz(clf, out_file = None,
                                feature_names = X_train.columns.values,
                                class_names= ['0','1','2','3'],
                                filled = True, rounded = True,
                                special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph. create_png())

# 将图像保存
graph.write_png("dtr_nice.png")

# 输出优化后训练集的分数。
Y_pred1 = clf.predict(X_train) # 模型预测
print(Y_pred1)

print("训练集",accuracy_score(Y_train, Y_pred1))

# 输出优化后测试集的分数。
Y_pred1 = clf.predict(X_test) # 模型预测
print(Y_pred1)

print("测试集",accuracy_score(Y_test, Y_pred1))