🔹 Phase 1 — Data Preprocessing

In this phase, we will build a preprocessing pipeline to clean and prepare the dataset for model training.

Input Dataset: C:\Users\turtlepeer\project1\AIOT_HW\HW3\sms_spam_no_header.csv

Output Dataset: C:\Users\turtlepeer\project1\AIOT_HW\HW3\sms_spam_clean.csv
File: preprocessing.py

Tasks:

Load the raw dataset.

Clean and normalize the text data (e.g., lowercase, remove punctuation, URLs, and stopwords).

Encode the labels (spam/ham).

Save the cleaned dataset for the next phase.

🔹 Phase 2 — Model Training and Prediction

This phase focuses on training and testing the Logistic Regression model.

Files:

train.py — trains the model using the cleaned dataset.

predict.py — loads the trained model and predicts new samples.

Tasks:

Split the cleaned dataset into training and testing sets.

Train a Logistic Regression model using TF-IDF features.

Evaluate the model with accuracy, precision, recall, and F1-score.

Save the trained model for later use in the app.

🔹 Phase 3 — Analysis and Visualization (Jupyter Notebook)

Create a complete Jupyter Notebook (HW3.ipynb) that integrates all previous steps (preprocessing → training → prediction).
The notebook will serve as a report and visualization document.

Requirements:

Each code cell must be preceded by a markdown cell written in Traditional Chinese, explaining what the code does.

Execute all cells to show real outputs.

Visualizations & Analysis:

Dataset overview (class distribution, token replacements in cleaned text)

Top tokens by class (spam vs ham)

Model performance metrics (accuracy, precision, recall, F1-score, confusion matrix)

ROC curve

Precision-Recall curve

Threshold sweep (precision/recall/F1 vs threshold)

Any additional insights or visualizations that enhance model understanding.

🔹 Phase 4 — Streamlit Deployment (App)

Develop a Streamlit web application (app.py) to demonstrate live spam detection.

Deployment goal:
Host the app on streamlit.app and make it interactive.

App Design:

Sidebar Parameters:

Dataset CSV path

Label column

Text column

Model directory

Text size

Random seed

Decision threshold

Main Dashboard Displays:

Dataset overview (class distribution, token replacements)

Top tokens by class (with graph)

Model performance (text summary + confusion matrix)

ROC and Precision-Recall curves

Threshold sweep visualization (precision/recall/F1 vs threshold)

Live Inference Section:

Buttons: “Use Spam Example” / “Use Ham Example”

Input box for custom message

“Predict” button

Display a dynamic spam probability graph after prediction.

🔹 Phase 5 — Final Report using OpenSpec

The final report will document how OpenSpec and AI Coding CLI were used throughout the project to:

Define the development phases and goals.

Automatically generate structured code and improve iterations.

Implement the logistic regression classifier with clear specifications.

Build, visualize, and deploy the final Streamlit app.

Deliverable:
A complete written report summarizing the entire development process, titled:

“Building a Spam Email Classification Streamlit App using OpenSpec”