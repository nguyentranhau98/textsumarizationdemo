from flask import Flask, render_template, request, jsonify
from ExtBert import ExtBert
from AbsBert import AbsBert
from gui_utils import make_ext_input_file, make_abs_input_file

extbert = ExtBert()
absbert = AbsBert()

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def complete():
    data = request.get_json()
    task = data['task']
    if (task == 'extractive'):
        make_ext_input_file(data['content'])
        res = extbert.ext_summarize()
    else:
        make_abs_input_file(data['content'])
        res = absbert.abs_summarize()
    # res = predict(data['content'], top_k=data['top_k'], top_p=data['top_p'], task=data['task'])
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008, debug=True)
