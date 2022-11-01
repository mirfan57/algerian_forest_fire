import pickle, bz2
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from app_logger import log
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

# Import Classification and Regression model file
clf_pickle = bz2.BZ2File('algerian_clf_model.pkl', 'rb')
reg_pickle = bz2.BZ2File('algerian_rf_model.pkl', 'rb')
#reg_pickle = open('algerian_forest_regression_model.pickle', 'rb')

#reg_pickle = bz2.BZ2File('Regression.pkl', 'rb')
clf_model = pickle.load(clf_pickle)
reg_model = pickle.load(reg_pickle)



# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')



# Route for Classification Model

@app.route('/predictClass', methods=['POST', 'GET'])
def predictClass():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Temperature= float(request.form['Temperature'])
            Relative_Humidity = int(request.form['RH'])
            Wind_Speed = int(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC= float(request.form['FFMC'])
            DMC= float(request.form['DMC'])
            DC = float(request.form['DC'])
            ISI= float(request.form['ISI'])

            features = [Temperature, Relative_Humidity, Wind_Speed, Rain, FFMC, DMC, DC, ISI]

            Float_features = [float(x) for x in features]
            final_features = [np.array(Float_features)]
            prediction = clf_model.predict(final_features)[0]

            log.info('Successfully Prediction completed for Classification model')

            if prediction == 0:
                text = 'Forest is Safe!'
            else:
                text = 'Forest is in Danger!'
            return render_template('index.html', prediction_text1="{} --- Chance of Fire is {}".format(text, prediction))
        except Exception as e:
            log.error('Input error, check input', e)
        return render_template('index.html', prediction_text1="Please check the input value.")


# Route for Regression Model


@app.route('/predictTemp', methods=['POST'])
def predictTemp():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            #Temperature=float(request.form['Temperature'])
            Relative_Humidity = int(request.form['RH'])
            Wind_Speed =int(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC= float(request.form['FFMC'])
            DMC= float(request.form['DMC'])
            DC= float(request.form['DC'])
            ISI= float(request.form['ISI'])

            features = [Relative_Humidity, Wind_Speed, Rain, FFMC, DMC, DC, ISI]

            Float_features = [float(x) for x in features]
            final_features = [np.array(Float_features)]
            prediction = reg_model.predict(final_features)[0]

            log.info('Successfully Prediction completed for Regression model')

            if prediction >= 35 and prediction < 43:
                return render_template('index.html', prediction_text2="WARNING!! Temperature is {} and quick action is required. Chances of fire is high.".format(prediction))
            elif prediction>=31 and prediction<35:
                return render_template('index.html', prediction_text2="Temperature is {:.2f} and the situation is ambiguous. Better to take precautions as there might be high fire chances".format(prediction))
            else:
                return render_template('index.html',
                                       prediction_text2="Temperature is {:.2f} and the situation is under control. Low fire chances".format(
                                           prediction))
        except Exception as e:
            log.error('Input error, check input', e)
        return render_template('index.html', prediction_text2="Please check the input value.")
            


# Run APP in Debug mode

if __name__ == "__main__":
    app.run(debug=True, port= 5000)
