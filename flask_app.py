from flask import Flask, render_template, request
import pickle
app = Flask(__name__)


@app.route('/result', methods=['GET', 'POST'])
def get_data():
    # отримання даних з форми
    pregnancies = int(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    bp = float(request.form['bp'])
    skin = float(request.form['skin'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['BMI'])
    pedigree = float(request.form['pedigree'])
    age = int(request.form['age'])
    data = [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]

    voting_ensemble = pickle.load(open('voting_ensemble.pkl', 'rb'))
    stacking_ensemble = pickle.load(open('stack_ensemble.pkl', 'rb'))
    sc = pickle.load(open('sc.pkl', 'rb'))
    mean = sc.mean_
    scale = sc.scale_

    scaled = []
    for i in range(len(data)):
        scaled.append((data[i]-mean[i])/scale[i])

    prediction1 = voting_ensemble.predict([scaled])
    prediction2 = stacking_ensemble.predict([scaled])
    return render_template("result.html", prediction1=prediction1, prediction2=prediction2, data=data)


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
