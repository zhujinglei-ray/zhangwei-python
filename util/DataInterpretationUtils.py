import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.Data import Data


# 准备 information value 的表
def prepare_info_value_table(data):
    information_value_table = pd.DataFrame(data.columns, columns=['variable'])
    information_value_table['IV'] = None
    namelist = ['FLAG_MOBIL', 'begin_month', 'dep_value', 'target', 'ID']

    for i in namelist:
        information_value_table.drop(information_value_table[information_value_table['variable'] == i].index,
                                     inplace=True)

    return information_value_table


# Define calc_iv function to calculate Information Value and WOE Value
# 定义 方法 去计算 信息价值 和 weight of evidence
# Calculate information value
def calculate_information_value(data_frame, feature, target, pr=False):
    lst = []

    data_frame[feature] = data_frame[feature].fillna("NULL")

    for i in range(data_frame[feature].nunique()):
        val = list(data_frame[feature].unique())[i]
        lst.append([feature,  # Variable
                    val,  # Value
                    data_frame[data_frame[feature] == val].count()[feature],  # All
                    data_frame[(data_frame[feature] == val) & (data_frame[target] == 0)].count()[feature],
                    # Good (think: Fraud == 0)
                    data_frame[(data_frame[feature] == val) & (data_frame[target] == 1)].count()[
                        feature]])  # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()
    print('This variable\'s information value is:', iv)
    print(data_frame[feature].value_counts())
    return iv, data


# 转换 虚拟变量（哑变量）
def convert_dummy(data_frame, feature, rank=0):
    pos = pd.get_dummies(data_frame[feature], prefix=feature)
    mode = data_frame[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest], axis=1, inplace=True)
    data_frame.drop([feature], axis=1, inplace=True)
    data_frame = data_frame.join(pos)
    return data_frame


# 得到分类
def get_category(data_frame, col, bins_num, labels, qcut=False):
    if qcut:
        local_df = pd.qcut(data_frame[col], q=bins_num, labels=labels)  # quantile cut
    else:
        local_df = pd.cut(data_frame[col], bins=bins_num, labels=labels)  # equal-length cut

    local_df = pd.DataFrame(local_df)
    name = 'gp' + '_' + col
    local_df[name] = local_df[col]
    data_frame = data_frame.join(local_df[name])
    data_frame[name] = data_frame[name].astype(object)
    return data_frame


# 输出 confusion 矩阵
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class DataDescriptionService:
    # 读取数据
    def __init__(self):
        dataSource = Data()
        self.data = dataSource.get_cleaned_data()
        self.info_value_table = prepare_info_value_table(self.data)

    # binary feature
    def describe_gender(self):
        data = self.data
        iv_table = self.info_value_table
        data['Gender'] = data['Gender'].replace(['F', 'M'], [0, 1])
        print(data['Gender'].value_counts())
        iv, calculated_data = calculate_information_value(data, 'Gender', 'target')
        iv_table.loc[iv_table['variable'] == 'Gender', 'IV'] = iv
        calculated_data.head()

    def describe_own_car(self):
        data = self.data
        iv_table = self.info_value_table
        data['Car'] = data['Car'].replace(['N', 'Y'], [0, 1])
        print(data['Car'].value_counts())
        iv, calculated_data = calculate_information_value(data, 'Car', 'target')
        iv_table.loc[iv_table['variable'] == 'Car', 'IV'] = iv
        calculated_data.head()

    def describe_own_house(self):
        data = self.data
        iv_table = self.info_value_table
        data['Reality'] = data['Reality'].replace(['N', 'Y'], [0, 1])
        print(data['Reality'].value_counts())
        iv, calculated_data = calculate_information_value(data, 'Reality', 'target')
        iv_table.loc[iv_table['variable'] == 'Reality', 'IV'] = iv
        calculated_data.head()

    def describe_own_phone(self):
        data = self.data
        iv_table = self.info_value_table
        data['phone'] = data['phone'].astype(str)
        print(data['phone'].value_counts(normalize=True, sort=False))
        data.drop(data[data['phone'] == 'nan'].index, inplace=True)
        iv, calculated_data = calculate_information_value(data, 'phone', 'target')
        iv_table.loc[iv_table['variable'] == 'phone', 'IV'] = iv
        calculated_data.head()

    def describe_own_email(self):
        data = self.data
        iv_table = self.info_value_table
        print(data['email'].value_counts(normalize=True, sort=False))
        data['email'] = data['email'].astype(str)
        iv, calculated_data = calculate_information_value(data, 'email', 'target')
        iv_table.loc[iv_table['variable'] == 'email', 'IV'] = iv
        calculated_data.head()

    def describe_own_working_phone(self):
        data = self.data
        iv_table = self.info_value_table
        data['wkphone'] = data['wkphone'].astype(str)
        data.drop(data[data['wkphone'] == 'nan'].index, inplace=True)
        iv, calculated_data = calculate_information_value(data, 'wkphone', 'target')
        iv_table.loc[iv_table['variable'] == 'wkphone', 'IV'] = iv
        calculated_data.head()
