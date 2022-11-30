import json
import os
# import easyocr

from flask import Flask, jsonify, request
from flask_cors import CORS
from excavator import excavator_recog
import webbrowser
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer, SignatureExpired, BadSignature

app = Flask(__name__, static_folder='./dist', static_url_path='/')
app.config['SECRET_KEY'] = 'trustedailab'
CORS(app)

max_user_num = 200
check_expired_user_interval = 20
check_expired_user_counter = 0
# reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory


def check_expired_user():
    global check_expired_user_counter
    check_expired_user_counter = check_expired_user_counter + 1
    if check_expired_user_counter >= check_expired_user_interval:
        check_expired_user_counter = 0
        with open('ids.json', 'r') as f:
            j = json.load(f)
            user_list = j[1]['user_list']
            for key in user_list:
                v_id = verify_auth_token(user_list[key])
                if v_id is None:
                    with open('ids.json', 'w') as f_w:
                        user_list.pop(key)
                        json.dump(j, f_w)
                    os.remove("out_excel/{}.xlsx".format(key))
                    print('user expired: {}'.format(key))
                    print('tokens remaining: {}'.format(len(user_list)))
                if v_id != key:
                    print('user id and token not identical!\nv_id: {}\nu_id: {}\ntoken: {}'
                          .format(v_id, key, user_list[key]))
                    # raise Exception('user id and token not identical!\nv_id: {}\nu_id: {}\ntoken: {}'
                    #                 .format(v_id, key, user_list[key]))


def allocate_id(json_var):
    newid = json_var[0]['new_id'] + 1
    for u in json_var[1]['user_list']:
        uid = int(u)
        if uid >= newid:
            newid = uid + 1
    if newid > 1000000:
        return 1
    return newid


# 获取token，有效时间1h
def generate_auth_token(expiration=3600):
    try:
        check_expired_user()
        with open('ids.json', 'r') as f:
            j = json.load(f)
            user_list = j[1]['user_list']
            if len(user_list) < max_user_num:
                user = allocate_id(j)
                j[0]['new_id'] = user
                s = Serializer(app.config['SECRET_KEY'], expires_in=expiration)
                t = s.dumps({'id': user})
                with open('ids.json', 'w') as f_w:
                    user_list[user] = str(t)
                    json.dump(j, f_w)
                return t
            return None
    except Exception as e:
        print("generate_auth_token(): {}".format(e))
        return None


# 解析token，确认用户身份
def verify_auth_token(token):
    s = Serializer(app.config['SECRET_KEY'])
    try:
        data = s.loads(token)
    except SignatureExpired:
        return None  # valid token, but expired
    except BadSignature:
        return None  # invalid token
    return data['id']


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


# @app.route('/api/recognition', methods=['POST'])
# def recognition():
#     files = request.files.to_dict()
#     h = request.headers
#     t = h['token']
#     proc_id = verify_auth_token(t)
#     if proc_id is None:
#         return "token invalid!", 401
#     res, mis = start_recognition(files, proc_id, reader)
#     if res is None:
#         return 'No legal screenshots!', 503
#     return jsonify({'res': res, 'mis': mis})


@app.route('/api/excavator', methods=['POST'])
def excavator():
    files = request.files.to_dict()
    h = request.headers
    t = h['token']
    proc_id = verify_auth_token(t)
    if proc_id is None:
        return "token invalid!", 401
    if len(files.keys()) != 1:
        return 'no legal video or more than 1 video!', 503
    vid_path = "temp_vid/{}.mp4".format(str(proc_id))
    list(files.values())[0].save(vid_path)
    res = excavator_recog(vid_path, proc_id, None)
    if res is None:
        return 'No legal video!', 503
    return jsonify({'res': res})


# @app.route('/api/getprog', methods=['GET'])
# def getprog():
#     p = request.args.to_dict()
#     t = p['token']
#     proc_id = verify_auth_token(t)
#     if proc_id is None:
#         return jsonify(-1)
#     prog = get_prog(proc_id)
#     return jsonify(prog)


@app.route('/api/gettoken', methods=['GET'])
def gettoken():
    t = generate_auth_token()
    if t is not None:
        return t
    else:
        return 'Too much users now', 503


@app.route('/api/getsum', methods=['GET'])
def getsum():
    with open('statis.json', 'r') as f:
        return jsonify(json.load(f)[0]['sum'])


@app.route('/api/destroytoken', methods=['DELETE'])
def destroytoken():
    p = request.args.to_dict()
    if len(p) == 0:
        return 'user not created'
    user = verify_auth_token(p['token'])
    try:
        with open('ids.json', 'w') as f:
            j = json.load(f)
            user_list = j[1]['user_list']
            user_list.pop(user)
            json.dump(j, f)
    except Exception as e:
        print(e)
    return 'OK'


# @app.route('/api/getexcel', methods=['GET'])
# def getexcel():
#     p = request.args.to_dict()
#     t = p['token']
#     proc_id = verify_auth_token(t)
#     if proc_id is None:
#         return "token invalid!", 401
#     f = get_excel(proc_id)
#     if f is not None:
#         return f
#     return jsonify('File has been deleted!')


@app.route('/<path:fallback>')
def fallback(fallback_url):  # Vue Router 的 mode 为 'hash' 时可移除该方法
    if fallback_url.startswith('css/') or fallback_url.startswith('js/') \
            or fallback_url.startswith('img/') or fallback_url == 'favicon.ico':
        return app.send_static_file(fallback_url)
    else:
        return app.send_static_file('index.html')


# @app.before_first_request
# def first_request():
#     app.model = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
#     return app.model


if __name__ == '__main__':
    # webbrowser.open('http://127.0.0.1:5000')
    # app.run(host="0.0.0.0", debug=False, port=5000)
    app.run(debug=False, port=5001)
