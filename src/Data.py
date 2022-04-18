import pandas as pd

# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
# 更改一下设置
pd.options.mode.chained_assignment = None


class Data:

    def __init__(self):
        self.applicationRecordPath = "data/application_record.csv"
        self.creditRecord = "data/credit_record.csv"

    # 读取用户数据 csv
    def read_application_record(self):
        application_data_source = pd.read_csv(self.applicationRecordPath, encoding='utf-8')
        return application_data_source

    # 读取信用数据
    def read_credit_record(self):
        credit_record_source = pd.read_csv(self.creditRecord, encoding='utf-8')
        return credit_record_source

    # 整理数据 找出所有用户的开户月份
    # 输出 整理完的数据 根据id 合并 数据集
    def prepare_by_merging_month_balance(self):
        application_data = self.read_application_record()
        credit_record = self.read_credit_record()
        start_month = pd.DataFrame(credit_record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
        start_month = start_month.rename(columns={'MONTHS_BALANCE': 'start_month'})
        new_data = pd.merge(application_data, start_month, how="left", on="ID")
        # merge to record data
        # print("合并后的数据的前 6 个数据 -- 可以删除")
        # print(new_data.head(6))
        return new_data

    # Generally, users in risk should be in 3%,
    # thus I choose users who overdue for more than 60 days as target risk users.
    # Those samples are marked as '1', else are '0'.
    # 通常来说 阀值 在 3% 左右 这里我标记 用户 超过 60天 没有还款的 作为 "坏" 用户 标记为 1 ，其他的标记为 0。

    # Record 里面 status的意思
    # 0: 1-29 days past due 1: 30-59 days past due 2: 60-89 days overdue 3: 90-119 days overdue 4: 120-149 days overdue
    # 5: Overdue or bad debts, write-offs for more than 150 days C: paid off that month X: No loan for the month
    def clean_data_by_labelling(self):
        credit_record = self.read_credit_record()
        merged_data = self.prepare_by_merging_month_balance();
        credit_record['dep_value'] = None
        credit_record['dep_value'][credit_record['STATUS'] == '2'] = 'Yes'
        credit_record['dep_value'][credit_record['STATUS'] == '3'] = 'Yes'
        credit_record['dep_value'][credit_record['STATUS'] == '4'] = 'Yes'
        credit_record['dep_value'][credit_record['STATUS'] == '5'] = 'Yes'

        cpunt = credit_record.groupby('ID').count()
        cpunt['dep_value'][cpunt['dep_value'] > 0] = 'Yes'
        cpunt['dep_value'][cpunt['dep_value'] == 0] = 'No'
        cpunt = cpunt[['dep_value']]
        new_data = pd.merge(merged_data, cpunt, how='inner', on='ID')
        new_data['target'] = new_data['dep_value']
        new_data.loc[new_data['target'] == 'Yes', 'target'] = 1
        new_data.loc[new_data['target'] == 'No', 'target'] = 0
        # cpunt['dep_value'].value_counts(normalize=True)
        # print("计算'坏' 用户的 值")
        # print(cpunt['dep_value'].value_counts())
        # print(new_data.head(6))
        return new_data

    # 重命名 并且 去除 空值
    def rename_data_frame(self):
        data = self.clean_data_by_labelling()
        data.rename(
            columns={'CODE_GENDER': 'gender', 'FLAG_OWN_CAR': 'Car', 'FLAG_OWN_REALTY': 'real_estate',
                     'CNT_CHILDREN': 'child_num', 'AMT_INCOME_TOTAL': 'inc',
                     'NAME_EDUCATION_TYPE': 'education_type', 'NAME_FAMILY_STATUS': 'family_type',
                     'NAME_HOUSING_TYPE': 'house_type', 'FLAG_EMAIL': 'email',
                     'NAME_INCOME_TYPE': 'inctp', 'FLAG_WORK_PHONE': 'work_phone',
                     'FLAG_PHONE': 'phone', 'CNT_FAM_MEMBERS': 'famsize',
                     'OCCUPATION_TYPE': 'occupation_type'
                     }, inplace=True)

        data = data.dropna()
        renamed_data = data.mask(data == 'NULL').dropna()
        # print("去掉 na 或者 空值")
        # print(renamed_data.head(6))
        return renamed_data

    def get_cleaned_data(self):
        return self.rename_data_frame()
