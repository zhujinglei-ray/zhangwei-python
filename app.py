from flask import Flask, request

app = Flask(__name__)


# flask 主页
@app.route('/')
def hello():
    return '欢迎来到我得 flask 应用 !'


# 准备跑程序
@app.route('/test', methods=['POST'])
def test():
    print(request.data)
    return "success"

if __name__ == '__main__':
    app.run(debug=True)
