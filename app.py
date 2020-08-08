# app.py
from flask import Flask
from flask import request
app = Flask(__name__)


@app.route("/cartoon", methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        return {
            'Image': data['Image']
        }
    else:
        return 'do not GET'


if __name__ == "__main__":
    app.run()
