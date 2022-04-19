from flask import Flask, request

import src.Predict as predict

app = Flask(__name__)


# flask 主页
@app.route('/')
def hello():
    return '欢迎来到我得 flask 应用 !'


# 准备跑程序
@app.route('/predict/lr', methods=['POST'])
def logistic_regression():
    print(request.data)
    res = predict.predict_with_logistic_regression(request.data)
    return res


@app.route('/predict/dt', methods=['POST'])
def decision_tree():
    print(request.data)
    res = predict.predict_with_decision_tree(request.data)
    return res


@app.route('/predict/rf', methods=['POST'])
def random_forest():
    print(request.data)
    res = predict.predict_with_random_forest(request.data)
    return res


@app.route('/predict/ann', methods=['POST'])
def ann():
    print(request.data)
    res = predict.predict_with_ann(request.data)
    return res


if __name__ == '__main__':
    app.run(debug=True)
