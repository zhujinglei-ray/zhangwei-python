本python 文件可以视为 算子，经过训练以后的模型 存在了 modellib 里面，可以供 其他程序调用


创建虚拟环境

`python3 -m venv env`

激活虚拟环境

`. env/bin/activate`

安装依赖
`pip3 install -r requirements.txt`

如果安装不成功（matplot）你需要升级pip的版本
`pip install --upgrade pip`



如果数据有更新 请 重新运行 main文件 去生成 新的模型

`python3 main.py`
