# 🏦 Loan Approval Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-0E7C7B)
![Streamlit](https://img.shields.io/badge/Streamlit-app-0B2540)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-0E4D64)
![License](https://img.shields.io/badge/License-MIT-green)

A Streamlit web app that predicts whether a loan application is likely to be **approved** or **rejected**, using a Support Vector Machine (SVM) trained on a small loan dataset.

## Features

- Clean, responsive interface with a **Predict**, **Model insights** and **About** tab
- Approval probability and model confidence for every prediction
- Loan-to-income ratio shown next to the result
- Dataset overview, test accuracy and confusion matrix
- Automatic data cleaning (comma-formatted income, invalid credit scores, duplicates, inconsistent labels)

## Inputs

| Field | Description |
|---|---|
| Annual income | In rupees (₹) |
| Loan amount | In rupees (₹) |
| Credit score | 300 to 900 |
| Years of experience | Work experience in years |
| Employment type | Salaried or Self-employed |

## Machine learning workflow

1. Load and clean `loan_approval.csv`.
2. Split into 80% training and 20% test data (stratified).
3. Fill missing values with the training median and scale features with `StandardScaler`.
4. Train an RBF-kernel SVM with calibrated probabilities.
5. Evaluate on the test set and predict for new applicants.

## Project structure

```text
loan-approval-prediction/
├── .streamlit/config.toml   # app theme
├── app.py                   # Streamlit app and model
├── loan_approval.csv        # dataset
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Run locally

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Deploy free on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Choose your repository, set the main file to `app.py`, and deploy.
4. Add the live link to this README.

## Upload to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Loan Approval Predictor"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

## Disclaimer

This project is for learning and demonstration only. Predictions are not real bank decisions. The dataset is small, so accuracy may vary and does not reflect real-world lending performance.

## License

Released under the MIT License. Replace `YOUR NAME` in `LICENSE` with your name.
