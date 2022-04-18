from flask import Flask, request
import src.Predict as predict
app = Flask(__name__)


# flask 主页
@app.route('/')
def hello():
    return '欢迎来到我得 flask 应用 !'


# 准备跑程序
@app.route('/test', methods=['POST'])
def test():
    print(request.data)
    res = predict.predict_with_logistic_regression(request.data)
    return res

if __name__ == '__main__':
    app.run(debug=True)
