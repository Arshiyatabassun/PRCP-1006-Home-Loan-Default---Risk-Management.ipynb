from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('home_loan_default_pred.joblib')

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            'AMT_INCOME_TOTAL': float(request.form['AMT_INCOME_TOTAL']),
            'AMT_CREDIT': float(request.form['AMT_CREDIT']),
            'AMT_ANNUITY': float(request.form['AMT_ANNUITY']),
            'AMT_GOODS_PRICE': float(request.form['AMT_GOODS_PRICE']),
            'DAYS_BIRTH': float(request.form['DAYS_BIRTH']),
            'DAYS_EMPLOYED': float(request.form['DAYS_EMPLOYED']),
            'EXT_SOURCE_2': float(request.form['EXT_SOURCE_2']),
            'EXT_SOURCE_3': float(request.form['EXT_SOURCE_3']),
            'CNT_CHILDREN': int(request.form['CNT_CHILDREN']),
            'CNT_FAM_MEMBERS': float(request.form['CNT_FAM_MEMBERS']),
            'REGION_POPULATION_RELATIVE': float(request.form['REGION_POPULATION_RELATIVE']),
            'CODE_GENDER': request.form['CODE_GENDER'],
            'NAME_EDUCATION_TYPE': request.form['NAME_EDUCATION_TYPE'],
            'NAME_FAMILY_STATUS': request.form['NAME_FAMILY_STATUS'],
            'FLAG_OWN_CAR': request.form['FLAG_OWN_CAR'],
            'FLAG_OWN_REALTY': request.form['FLAG_OWN_REALTY'],

            'CREDIT_INCOME_RATIO': float(request.form['AMT_CREDIT']) / float(request.form['AMT_INCOME_TOTAL']),
            'ANNUITY_INCOME_RATIO': float(request.form['AMT_ANNUITY']) / float(request.form['AMT_INCOME_TOTAL']),
            'AGE_YEARS': -float(request.form['DAYS_BIRTH']) / 365,
            'EMPLOYED_YEARS': -float(request.form['DAYS_EMPLOYED']) / 365 if float(request.form['DAYS_EMPLOYED']) < 0 else 0,

            'BUREAU_SK_ID_BUREAU_nunique': float(request.form['BUREAU_SK_ID_BUREAU_nunique']),
            'BUREAU_AMT_CREDIT_SUM_mean': float(request.form['BUREAU_AMT_CREDIT_SUM_mean']),
            'BUREAU_AMT_CREDIT_SUM_DEBT_mean': float(request.form['BUREAU_AMT_CREDIT_SUM_DEBT_mean']),
            'BUREAU_CREDIT_DAY_OVERDUE_max': float(request.form['BUREAU_CREDIT_DAY_OVERDUE_max']),

            'PREVIOUS_SK_ID_PREV_nunique': float(request.form['PREVIOUS_SK_ID_PREV_nunique']),
            'PREVIOUS_AMT_APPLICATION_mean': float(request.form['PREVIOUS_AMT_APPLICATION_mean']),
            'PREVIOUS_AMT_CREDIT_mean': float(request.form['PREVIOUS_AMT_CREDIT_mean']),
            'PREVIOUS_AMT_ANNUITY_mean': float(request.form['PREVIOUS_AMT_ANNUITY_mean']),

            'POS_SK_ID_PREV_nunique': float(request.form['POS_SK_ID_PREV_nunique']),
            'POS_CNT_INSTALMENT_mean': float(request.form['POS_CNT_INSTALMENT_mean']),
            'POS_SK_DPD_max': float(request.form['POS_SK_DPD_max']),

            'CC_SK_ID_PREV_nunique': float(request.form['CC_SK_ID_PREV_nunique']),
            'CC_AMT_BALANCE_mean': float(request.form['CC_AMT_BALANCE_mean']),
            'CC_AMT_CREDIT_LIMIT_ACTUAL_mean': float(request.form['CC_AMT_CREDIT_LIMIT_ACTUAL_mean']),
            'CC_SK_DPD_max': float(request.form['CC_SK_DPD_max']),

            'INST_SK_ID_PREV_nunique': float(request.form['INST_SK_ID_PREV_nunique']),
            'INST_AMT_INSTALMENT_mean': float(request.form['INST_AMT_INSTALMENT_mean']),
            'INST_AMT_PAYMENT_mean': float(request.form['INST_AMT_PAYMENT_mean']),
            'INST_DAYS_ENTRY_PAYMENT_mean': float(request.form['INST_DAYS_ENTRY_PAYMENT_mean'])
        }

        input_df = pd.DataFrame([features])

        probability = model.predict_proba(input_df)[0, 1]
        prediction = int(probability >= 0.50)

        if prediction == 1:
            result_text = f'⚠️ Higher estimated default risk. Probability: {probability:.2%}'
            result_class = 'bg-red-50 text-red-700 border-red-200'
        else:
            result_text = f'✅ Lower estimated default risk. Probability: {probability:.2%}'
            result_class = 'bg-emerald-50 text-emerald-700 border-emerald-200'

        return render_template(
            'index.html',
            prediction_text=result_text,
            result_class=result_class
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            result_class='bg-amber-50 text-amber-700 border-amber-200'
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
