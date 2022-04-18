from src.DataInterpretationUtils import DataDescriptionService

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
