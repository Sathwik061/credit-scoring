import gradio as gr
import joblib
import pandas as pd


# =======================
# LOAD MODEL + FEATURES
# =======================

log_model = joblib.load("logistic_model.pkl")
feature_names = joblib.load("model_features.pkl")


# =======================
# CATEGORY MAPPINGS
# =======================

checking_map = {
    "No account": "A11",
    "Balance < 0": "A12",
    "Balance 0–200": "A13",
    "Balance ≥ 200": "A14"
}

cred_hist_map = {
    "No credit taken": "A30",
    "All credits paid properly": "A31",
    "Existing credits paid": "A32",
    "Delay in paying": "A33",
    "Critical account": "A34",
}

purpose_map = {
    "New car": "new car",
    "Used car": "used car",
    "Radio / TV": "radio/tv",
    "Furniture": "furniture",
    "Business": "business",
    "Education": "education",
    "Repairs": "repairs",
    "Vacation": "vacation"
}

savings_map = {
    "No savings": "A61",
    "< 100": "A62",
    "100 – 500": "A63",
    "500 – 1000": "A64",
    "≥ 1000": "A65",
}

employment_map = {
    "Unemployed": "A71",
    "< 1 year": "A72",
    "1–4 years": "A73",
    "4–7 years": "A74",
    "≥ 7 years": "A75",
}

gender_map = {
    "Male – single": "A93",
    "Male – married": "A94",
    "Female – single/divorced": "A92",
}

property_map = {
    "Real estate": "A121",
    "Building society savings": "A122",
    "Car": "A123",
    "No property": "A124",
}

housing_map = {
    "Own": "own",
    "Rent": "rent",
    "Free": "free",
}

job_map = {
    "Unemployed / unskilled": "A171",
    "Unskilled": "A172",
    "Skilled": "A173",
    "Management / self-employed": "A174",
}


# =======================
# PREDICT FUNCTION
# =======================

def predict_credit(
    checking_acc_status,
    duration,
    cred_hist,
    purpose,
    loan_amt,
    saving_acc_bonds,
    present_employment_since,
    installment_rate,
    personal_stat_gender,
    present_residence_since,
    property,
    age,
    housing,
    num_curr_loans,
    job,
):

    data = {
        "checking_acc_status": checking_map[checking_acc_status],
        "duration": duration,
        "cred_hist": cred_hist_map[cred_hist],
        "purpose": purpose_map[purpose],
        "loan_amt": loan_amt,
        "saving_acc_bonds": savings_map[saving_acc_bonds],
        "present_employment_since": employment_map[present_employment_since],
        "installment_rate": installment_rate,
        "personal_stat_gender": gender_map[personal_stat_gender],
        "present_residence_since": present_residence_since,
        "property": property_map[property],
        "age": age,
        "housing": housing_map[housing],
        "num_curr_loans": num_curr_loans,
        "job": job_map[job],
    }

    df = pd.DataFrame([data])

    # match training preprocessing
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_names, fill_value=0)

    # -------- probability instead of predict() --------
    prob = log_model.predict_proba(df)[0][1]

    # tune threshold if required
    threshold = 0.40

    label = "Good Credit" if prob >= threshold else "Bad Credit"

    return f"Logistic Regression Prediction: {label} (probability = {prob:.2f})"


# =======================
# GRADIO UI
# =======================

inputs = [
    gr.Dropdown(list(checking_map.keys()), label="Checking Account Status"),
    gr.Number(label="Duration (months)"),
    gr.Dropdown(list(cred_hist_map.keys()), label="Credit History"),
    gr.Dropdown(list(purpose_map.keys()), label="Purpose"),
    gr.Number(label="Loan Amount"),
    gr.Dropdown(list(savings_map.keys()), label="Savings Account / Bonds"),
    gr.Dropdown(list(employment_map.keys()), label="Employment Since"),
    gr.Number(label="Installment Rate"),
    gr.Dropdown(list(gender_map.keys()), label="Gender / Personal Status"),
    gr.Number(label="Residence Since (years)"),
    gr.Dropdown(list(property_map.keys()), label="Property"),
    gr.Number(label="Age"),
    gr.Dropdown(list(housing_map.keys()), label="Housing"),
    gr.Number(label="Number of Current Loans"),
    gr.Dropdown(list(job_map.keys()), label="Job"),
]

outputs = gr.Textbox(label="Model Prediction")

app = gr.Interface(
    fn=predict_credit,
    inputs=inputs,
    outputs=outputs,
    title="Credit Scoring Prediction App",
    description="Credit scoring using Logistic Regression only.",
)

app.launch()
