import joblib
import json
from pandas import DataFrame


def predict_with_logistic_regression(data):
    model = joblib.load("modellib/logistic_regression_model.pkl")
    jsonData = json.loads(data)
    # print(type(jsonData))

    print(jsonData.get("gender"))

    # test_data = {'gender': 0.0, 'real_estate': 1.0, 'child_num_1': 1.0, 'child_num_2More': 0.0, 'work_phone': 0.0,
    #              'gp_age_high': 0.0, 'gp_age_highest': 0.0, 'gp_age_low': 0.0,
    #              'gp_age_lowest': 0.0, 'gp_work_time_high': 0.0, 'gp_work_time_highest': 0.0,
    #              'gp_work_time_low': 0.0, 'gp_work_time_medium': 0.0, 'occupation_type_hightecwk': 0.0,
    #              'occupation_type_officewk': 0.0, 'famsizegp_1': 0.0, 'famsizegp_3more': 1.0,
    #              'house_type_Co-op apartment': 0.0, 'house_type_Municipal apartment': 0.0,
    #              'house_type_Office apartment': 0.0, 'house_type_Rented apartment': 0.0,
    #              'house_type_With parents': 0.0, 'education_type_Higher education': 0.0,
    #              'education_type_Incomplete higher': 0.0, 'education_type_Lower secondary': 0.0,
    #              'family_type_Civil marriage': 0.0,
    #              'family_type_Separated': 1, 'family_type_Single / not married': 0, 'family_type_Widow': 0}

    df = DataFrame(jsonData, index=[0])
    res = model.predict_proba(df)
    print(res[0][0])
    return str(res[0][0])
