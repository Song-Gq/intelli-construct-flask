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
$ conda create -n ocr-sample-flask python=3.6
$ conda activate ocr-sample-flask
$ pip install easyocr
$ pip install flask flask_cors xlwt
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

```shell
$ cd ocr-sample-flask
$ python main.py
```

通过浏览器访问 http://127.0.0.1:5000

- 初次识别可能会自动下载模型，下载完成后可能会需要一些时间进行处理
- 输出excel文件在代码根目录下

#### 部署

- 请自行配置`Nginx`，`/etc/nginx/conf.d/`

  ```nginx
  server {
      listen 8888; #外部HTTP访问端口
      server_name localhost;
      location / {
          include uwsgi_params;
          uwsgi_pass 127.0.0.1:5000; #uwsgi端口
       }
  }
  ```

- `uwsgi`配置文件已包含在代码根目录，端口5000

  ```ini
  [uwsgi]
  # 使用nginx时为socket而非http
  socket = 127.0.0.1:5000 
  # conda environment
  home=/home/veocw/anaconda3/envs/ocr-sample-flask
  # Flask script
  wsgi-file=/mnt/data/sgq/ocr/ocr-sample-flask/main.py
  callable=app
  # 根据GPU Memory调整Process数
  processes=2
  threads=16
  buffer-size=32768
  master=true
  stats=/mnt/data/sgq/ocr/ocr-sample-flask/uwsgi.status
  pidfile=/mnt/data/sgq/ocr/ocr-sample-flask/uwsgi.pid
  lazy=true
  ```

- 然后运行，通过浏览器访问http://127.0.0.1:8888/

```shell
$ sudo nginx
$ uwsgi config.ini
$ sudo nginx -s reload
```

#### (Optional) 生成可执行文件

```shell
$ pip install pyinstaller
$ pyinstaller -D main.py
```

