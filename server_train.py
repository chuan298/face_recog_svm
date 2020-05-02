from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from datetime import datetime
import extract_embeddings
import train_model
import recognize

app = Flask(__name__)
CORS(app)
@app.route("/")
def main():
    return "Duyen xinh gai"



@app.route("/train", methods = ['POST'])
def train():
    print("train", flush=True)
    try:
        request_json = request.get_json()
        data = {
            'image': request_json.get('image'),
            'label': request_json.get('label')
        }
        print("dadsa:",data)
        is_extract = extract_embeddings.extract_embedding(data['image'], data['label'])
        is_train = train_model.train_model()
        return jsonify({"is_extract": is_extract, "is_train": is_train})

    except Exception as e:
        print(e, flush=True)

@app.route("/recog", methods = ['POST'])
def recog():
    print("recog", flush=True)
    try:
        request_json = request.get_json()
        data = {
            'image': request_json.get('image')
        }
        response = recognize.recognize(data['image'])
        return jsonify(response)
    except Exception as e:
        print(e, flush=True)

if __name__ == "__main__":
    http_server = WSGIServer(('localhost', 1602), app)
    http_server.serve_forever()