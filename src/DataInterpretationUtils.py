import numpy as np
import pandas as pd

from src.Data import Data


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
def get_category(df, col, binsnum, labels, qcut=False):
    if qcut:
        localdf = pd.qcut(df[col], q=binsnum, labels=labels)  # quantile cut
    else:
        localdf = pd.cut(df[col], bins=binsnum, labels=labels)  # equal-length cut

    localdf = pd.DataFrame(localdf)
    name = 'gp' + '_' + col
    localdf[name] = localdf[col]
    df = df.join(localdf[name])
    df[name] = df[name].astype(object)
    return df


class DataDescriptionService:
    # 读取数据
    def __init__(self):
        dataSource = Data()
        self.data = dataSource.get_cleaned_data()
        self.info_value_table = prepare_info_value_table(self.data)

    # binary feature
    def describe_binary_value(self):
        self.describe_gender()
        self.describe_own_car()
        self.describe_own_house()
        self.describe_own_email()
        self.describe_own_phone()
        self.describe_own_working_phone()

    def describe_gender(self):
        iv_table = self.info_value_table
        self.data['gender'] = self.data['gender'].replace(['F', 'M'], [0, 1])
        print(self.data['gender'].value_counts())
        iv, calculated_data = calculate_information_value(self.data, 'gender', 'target')
        iv_table.loc[iv_table['variable'] == 'gender', 'IV'] = iv
        print(calculated_data.head())

    def describe_own_car(self):
        iv_table = self.info_value_table
        self.data['Car'] = self.data['Car'].replace(['N', 'Y'], [0, 1])
        print(self.data['Car'].value_counts())
        iv, calculated_data = calculate_information_value(self.data, 'Car', 'target')
        iv_table.loc[iv_table['variable'] == 'Car', 'IV'] = iv
        print()
        calculated_data.head()

    def describe_own_house(self):
        iv_table = self.info_value_table
        self.data['real_estate'] = self.data['real_estate'].replace(['N', 'Y'], [0, 1])
        print(self.data['real_estate'].value_counts())
        iv, calculated_data = calculate_information_value(self.data, 'real_estate', 'target')
        iv_table.loc[iv_table['variable'] == 'real_estate', 'IV'] = iv
        print(calculated_data.head())

    def describe_own_phone(self):
        iv_table = self.info_value_table
        self.data['phone'] = self.data['phone'].astype(str)
        print(self.data['phone'].value_counts(normalize=True, sort=False))
        self.data.drop(self.data[self.data['phone'] == 'nan'].index, inplace=True)
        iv, calculated_data = calculate_information_value(self.data, 'phone', 'target')
        iv_table.loc[iv_table['variable'] == 'phone', 'IV'] = iv
        print(calculated_data.head())

    def describe_own_email(self):
        iv_table = self.info_value_table
        print(self.data['email'].value_counts(normalize=True, sort=False))
        self.data['email'] = self.data['email'].astype(str)
        iv, calculated_data = calculate_information_value(self.data, 'email', 'target')
        iv_table.loc[iv_table['variable'] == 'email', 'IV'] = iv
        print(calculated_data.head())

    def describe_own_working_phone(self):
        iv_table = self.info_value_table
        self.data['work_phone'] = self.data['work_phone'].astype(str)
        self.data.drop(self.data[self.data['work_phone'] == 'nan'].index, inplace=True)
        iv, calculated_data = calculate_information_value(self.data, 'work_phone', 'target')
        iv_table.loc[iv_table['variable'] == 'work_phone', 'IV'] = iv
        print(calculated_data.head())

    # Continuous Variables
    def describe_continuous_value(self):
        self.describe_child_num()
        self.describe_annual_income()
        self.describe_age()
        self.describe_working_years()
        self.describe_family_size()

    # def convert_continuous_to_dummy(self):

    def describe_child_num(self):
        iv_table = self.info_value_table
        self.data.loc[self.data['child_num'] >= 2, 'child_num'] = '2More'
        print(self.data['child_num'].value_counts(sort=False))
        iv, calculated_data = calculate_information_value(self.data, 'child_num', 'target')
        iv_table.loc[iv_table['variable'] == 'child_num', 'IV'] = iv

        self.data = convert_dummy(self.data, 'child_num')
        print(calculated_data.head())

    def describe_annual_income(self):
        iv_table = self.info_value_table
        self.data['inc'] = self.data['inc'].astype(object)
        self.data['inc'] = self.data['inc'] / 10000
        print(self.data['inc'].value_counts(bins=10, sort=False))
        self.data['inc'].plot(kind='hist', bins=50, density=True)

        # 这里有bug 就先不用 qcut做 直接cut吧， information value 差了 1倍 从 0.002 降到了 0.001
        self.data = get_category(self.data, 'inc', 5, ["very low", "low", "medium", "high", "very high"])
        iv, calculated_data = calculate_information_value(self.data, 'gp_inc', 'target')
        iv_table.loc[iv_table['variable'] == 'inc', 'IV'] = iv
        self.data = convert_dummy(self.data, 'gp_inc')
        # print(calculated_data.head())

    def describe_age(self):
        iv_table = self.info_value_table
        self.data['age'] = -(self.data['DAYS_BIRTH']) // 365
        print(self.data['age'].value_counts(bins=10, normalize=True, sort=False))
        self.data['age'].plot(kind='hist', bins=20, density=True)

        self.data = get_category(self.data, 'age', 5, ["lowest", "low", "medium", "high", "highest"])
        iv, data = calculate_information_value(self.data, 'gp_age', 'target')
        iv_table.loc[iv_table['variable'] == 'DAYS_BIRTH', 'IV'] = iv
        print(data.head())
        self.data = convert_dummy(self.data, 'gp_age')

    def describe_working_years(self):
        iv_table = self.info_value_table
        self.data['work_time'] = -(self.data['DAYS_EMPLOYED']) // 365
        self.data[self.data['work_time'] < 0] = np.nan  # replace by na
        self.data['DAYS_EMPLOYED']
        self.data['work_time'].fillna(self.data['work_time'].mean(), inplace=True)  # replace na by mean
        self.data['work_time'].plot(kind='hist', bins=20, density=True)

        self.data = get_category(self.data, 'work_time', 5, ["lowest", "low", "medium", "high", "highest"])
        iv, data = calculate_information_value(self.data, 'gp_work_time', 'target')
        iv_table.loc[iv_table['variable'] == 'DAYS_EMPLOYED', 'IV'] = iv
        print(data.head())

        self.data = convert_dummy(self.data, 'gp_work_time')

    def describe_family_size(self):
        iv_table = self.info_value_table
        self.data['famsize'].value_counts(sort=False)

        self.data['famsize'] = self.data['famsize'].astype(int)
        self.data['famsizegp'] = self.data['famsize']
        self.data['famsizegp'] = self.data['famsizegp'].astype(object)
        self.data.loc[self.data['famsizegp'] >= 3, 'famsizegp'] = '3more'
        iv, data = calculate_information_value(self.data, 'famsizegp', 'target')
        iv_table.loc[iv_table['variable'] == 'famsize', 'IV'] = iv
        print(data.head())
        self.data = convert_dummy(self.data, 'famsizegp')

    def describe_categorical_variable(self):
        self.describe_income_type()
        self.describe_occupation_type()
        self.describe_house_type()
        self.describe_education()
        self.describe_marriage_condition()

    def describe_income_type(self):
        iv_table = self.info_value_table
        print(self.data['inctp'].value_counts(sort=False))
        print(self.data['inctp'].value_counts(normalize=True, sort=False))
        self.data.loc[self.data['inctp'] == 'Pensioner', 'inctp'] = 'State servant'
        self.data.loc[self.data['inctp'] == 'Student', 'inctp'] = 'State servant'
        iv, data = calculate_information_value(self.data, 'inctp', 'target')
        iv_table.loc[iv_table['variable'] == 'inctp', 'IV'] = iv
        data.head()
        self.data = convert_dummy(self.data, 'inctp')

    def describe_occupation_type(self):
        iv_table = self.info_value_table
        self.data.loc[(self.data['occupation_type'] == 'Cleaning staff') | (self.data['occupation_type'] == 'Cooking staff') | (
                self.data['occupation_type'] == 'Drivers') | (self.data['occupation_type'] == 'Laborers') | (
                              self.data['occupation_type'] == 'Low-skill Laborers') | (
                              self.data['occupation_type'] == 'Security staff') | (
                              self.data['occupation_type'] == 'Waiters/barmen staff'), 'occupation_type'] = 'Laborwk'
        self.data.loc[(self.data['occupation_type'] == 'Accountants') | (self.data['occupation_type'] == 'Core staff') | (
                self.data['occupation_type'] == 'HR staff') | (self.data['occupation_type'] == 'Medicine staff') | (
                              self.data['occupation_type'] == 'Private service staff') | (
                              self.data['occupation_type'] == 'Realty agents') | (self.data['occupation_type'] == 'Sales staff') | (
                              self.data['occupation_type'] == 'Secretaries'), 'occupation_type'] = 'officewk'
        self.data.loc[(self.data['occupation_type'] == 'Managers') | (self.data['occupation_type'] == 'High skill tech staff') | (
                self.data['occupation_type'] == 'IT staff'), 'occupation_type'] = 'hightecwk'
        print(self.data['occupation_type'].value_counts())
        iv, data = calculate_information_value(self.data, 'occupation_type', 'target')
        iv_table.loc[iv_table['variable'] == 'occupation_type', 'IV'] = iv
        data.head()

        self.data = convert_dummy(self.data, 'occupation_type')

    def describe_house_type(self):
        iv_table = self.info_value_table
        iv, data = calculate_information_value(self.data, 'house_type', 'target')
        iv_table.loc[iv_table['variable'] == 'house_type', 'IV'] = iv
        data.head()
        self.data = convert_dummy(self.data, 'house_type')

    def describe_education(self):
        iv_table = self.info_value_table
        self.data.loc[self.data['education_type'] == 'Academic degree', 'education_type'] = 'Higher education'
        iv, data = calculate_information_value(self.data, 'education_type', 'target')
        iv_table.loc[iv_table['variable'] == 'education_type', 'IV'] = iv
        data.head()

        self.data = convert_dummy(self.data, 'education_type')

    def describe_marriage_condition(self):
        iv_table = self.info_value_table
        self.data['family_type'].value_counts(normalize=True, sort=False)

        iv, data = calculate_information_value(self.data, 'family_type', 'target')
        iv_table.loc[iv_table['variable'] == 'family_type', 'IV'] = iv
        data.head()

        self.data = convert_dummy(self.data, 'family_type')

    # 最终显示 信息价值
    def show_information_value(self):
        iv_table = self.info_value_table
        ivtable = iv_table.sort_values(by='IV', ascending=False)
        ivtable.loc[ivtable['variable'] == 'DAYS_BIRTH', 'variable'] = 'agegp'
        ivtable.loc[ivtable['variable'] == 'DAYS_EMPLOYED', 'variable'] = 'work_timegp'
        ivtable.loc[ivtable['variable'] == 'inc', 'variable'] = 'incgp'
        print(ivtable)

    def get_data_for_modeling(self):
        return self.data
