from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        feature = [float(i) for i in request.form.values()]
        feature = [np.array(feature)]
        y_pred = model.predict(feature)

    return render_template('index.html', predict_price="House price is {}".format(y_pred))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)