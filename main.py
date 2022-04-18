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

model.simple_ann()
