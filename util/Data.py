import pandas as pd


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
        print("合并后的数据的前 6 个数据 -- 可以删除")
        print(new_data.head(6))
        return new_data

    # Generally, users in risk should be in 3%,
    # thus I choose users who overdue for more than 60 days as target risk users.
    # Those samples are marked as '1', else are '0'.
    # 通常来说 阀值 在 3% 左右 这里我标记 用户 超过 60天 没有还款的 作为 "坏" 用户 标记为 1 ，其他的标记为 0。
    def clean_data_by_labelling(self):
        credit_record = self.read_credit_record()


