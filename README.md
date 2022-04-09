## 核酸截图OCR

#### 依赖

```shell
conda create -n ocr-sample-flask python=3.6
conda activate ocr-sample-flask
pip install easyocr
pip install flask flask_cors xlwt

(Optional)
cd EasyOCR-master
python setup.py install
```

#### 运行

```
cd ocr-sample-flask
python main.py
```

浏览器访问 http://127.0.0.1:5000

初次识别可能会自动下载模型，下载完成后可能会需要一些时间进行处理

输出excel文件在代码根目录下

#### 生成可执行文件

```
pip install pyinstaller
pyinstaller -D main.py
```

