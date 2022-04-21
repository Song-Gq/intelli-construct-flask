## 核酸截图OCR

Prerequisite: `Conda`, 

If wants GPU enabled: `Cuda`, `CuDNN` `Pytorch(with cuda)`

Front-end: `Vue2` https://github.com/Song-Gq/shanghai-nucleic-acid-ocr-vue

Back-end: `Flask` https://github.com/Song-Gq/shanghai-nucleic-acid-ocr

OCR-model: https://github.com/JaidedAI/EasyOCR

#### TODOLIST

- [ ] enable deployment mode
- [x] client token
- [x] disable excess file size and types
- [x] export to excel interface
- [x] delete expired excel files on server
- [x] web table for needing attention samples
- [ ] recognition result roboustness problem
- [ ] running efficiency on cpu
- [x] list.index() to fuzz_index()
- [x] choose folder to upload
- [ ] batch delete

#### 依赖

```shell
conda create -n ocr-sample-flask python=3.6
conda activate ocr-sample-flask
pip install easyocr
pip install flask flask_cors xlwt
```

#### 识别模型

将`.EasyOCR`解压缩并放至`~/.EasyOCR`

- For Windows

  `C:\Users\${用户名}\.EasyOCR`

- For Linux

  `/home/${用户名}/.EasyOCR`

共包含两个模型文件

- `~/.EasyOCR/model/craft_mlt_25k.pth`

  https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip

- `~/.EasyOCR/model/zh_sim_g2.pth`

  https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip

#### 运行

```
cd ocr-sample-flask
python main.py
```

通过浏览器访问 http://127.0.0.1:5000

- 初次识别可能会自动下载模型，下载完成后可能会需要一些时间进行处理

- 输出excel文件在代码根目录下

#### (Optional) 生成可执行文件

```
pip install pyinstaller
pyinstaller -D main.py
```

