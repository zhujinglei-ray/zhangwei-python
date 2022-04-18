import joblib

from src.DataInterpretationUtils import DataDescriptionService
from src.Model import Model

# 程序 入口
# 读取清理好的数据
# data = Data()
# data.get_cleaned_data()

# 数据描述
descriptionService = DataDescriptionService()
descriptionService.describe_binary_value()
descriptionService.describe_continuous_value()
descriptionService.describe_categorical_variable()
descriptionService.show_information_value()

# 算法
data = descriptionService.get_data_for_modeling();
print(data.columns)
# 逻辑 回归
model = Model(data)
logistic_regression_model = model.logistic_regression()
# 简单决策树
decision_tree = model.simple_decision_tree()
# 随机森林
random_forest = model.random_forest()
# 简单神经网络
ann = model.simple_ann()

joblib.dump(logistic_regression_model, 'modellib/logistic_regression_model.pkl')
joblib.dump(decision_tree, 'modellib/decision_tree.pkl')
joblib.dump(random_forest, 'modellib/random_forest.pkl')
joblib.dump(ann, 'modellib/ann.pkl')

# cf3 = joblib.load('filename.pkl')
# print(cf3.predict_proba(df))

# 创建测试数据 到时候 要用json 转过来
# test_data = {'Gender': 0.0, 'Reality': 1.0, 'ChldNo_1': 1.0, 'ChldNo_2More': 0.0, 'wkphone': 0.0,
#              'gp_Age_high': 0.0, 'gp_Age_highest': 0.0, 'gp_Age_low': 0.0,
#              'gp_Age_lowest': 0.0, 'gp_worktm_high': 0.0, 'gp_worktm_highest': 0.0,
#              'gp_worktm_low': 0.0, 'gp_worktm_medium': 0.0, 'occyp_hightecwk': 0.0,
#              'occyp_officewk': 0.0, 'famsizegp_1': 0.0, 'famsizegp_3more': 0.0,
#              'houtp_Co-op apartment': 0.0, 'houtp_Municipal apartment': 0.0,
#              'houtp_Office apartment': 0.0, 'houtp_Rented apartment': 0.0,
#              'houtp_With parents': 0.0, 'edutp_Higher education': 0.0,
#              'edutp_Incomplete higher': 0.0, 'edutp_Lower secondary': 0.0, 'famtp_Civil marriage': 0.0,
#              'famtp_Separated': 0.0, 'famtp_Single / not married': 0, 'famtp_Widow': 0}
# df = DataFrame(test_data, index=[0])

# 用pickle 读写模型
# s = pickle.dumps(lr)
# print(s)
# cf2 = pickle.loads(s)
# print(cf2.predict_proba(df))
