from util.DataInterpretationUtils import DataDescriptionService

# 程序 入口
# 读取清理好的数据
# data = Data()
# data.get_cleaned_data()

descriptionService = DataDescriptionService()

descriptionService.describe_gender()
descriptionService.describe_own_email()
descriptionService.describe_own_car()
descriptionService.describe_own_house()
descriptionService.describe_own_working_phone()
