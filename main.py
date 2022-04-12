from flask import Flask, jsonify, request
from flask_cors import CORS
from start_recognition import start_recognition, get_prog
import webbrowser

app = Flask(__name__, static_folder='./dist', static_url_path='/')
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/api/recognition', methods=['POST'])
def recognition():
    files = request.files.to_dict()
    res, mis, query_id = start_recognition(files)
    return jsonify({'res': res, 'mis': mis, 'query_id': query_id})


@app.route('/api/getprog', methods=['GET'])
def getprog():
    p = request.args.to_dict()
    proc_id = p['proc_ID']
    prog = get_prog(proc_id)
    return jsonify(prog)


@app.route('/<path:fallback>')
def fallback(fallback_url):  # Vue Router 的 mode 为 'hash' 时可移除该方法
    if fallback_url.startswith('css/') or fallback_url.startswith('js/') \
            or fallback_url.startswith('img/') or fallback_url == 'favicon.ico':
        return app.send_static_file(fallback_url)
    else:
        return app.send_static_file('index.html')


if __name__ == '__main__':
    # webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True)
