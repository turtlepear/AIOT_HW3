## ADDED Requirements

### Requirement: Data Preprocessing Pipeline
The system SHALL provide a preprocessing pipeline to clean the raw SMS spam dataset and persist a cleaned dataset for modeling.

#### Scenario: Clean and persist dataset
- **WHEN** a CSV path is provided for the raw dataset
- **THEN** the system loads the data, normalizes text (e.g., lowercase, remove punctuation/URLs/stopwords), encodes labels, and saves a cleaned CSV to a specified path

### Requirement: Model Training and Evaluation (TF-IDF + Logistic Regression)
The system SHALL train a Logistic Regression model using TF-IDF features and persist the trained artifacts.

#### Scenario: Train, evaluate, and save artifacts
- **WHEN** the cleaned dataset path is provided
- **THEN** the system splits data into train/test, trains the model, evaluates accuracy/precision/recall/F1, and saves the model and vectorizer for later inference

### Requirement: Analysis and Visualization Notebook
The system SHALL include a Jupyter notebook that documents preprocessing, training, and prediction with Traditional Chinese explanations.

#### Scenario: Executable end-to-end notebook with Traditional Chinese markdown
- **WHEN** the notebook `HW3.ipynb` is opened
- **THEN** every code cell is preceded by a Traditional Chinese markdown cell explaining intent, and executing all cells completes without error producing required visualizations and metrics

### Requirement: Streamlit App for Live Spam Detection
The system SHALL provide a Streamlit app with configurable parameters, dataset/model overviews, and live inference.

#### Scenario: Interactive dashboard and inference
- **WHEN** the app is launched with sidebar parameters (dataset path, label/text columns, model directory, text size, seed, threshold)
- **THEN** the app displays dataset overview, top tokens, model metrics (including confusion matrix), ROC and PR curves, threshold sweep, and supports example and custom inputs with predicted probability visualization

### Requirement: Final Report using OpenSpec
The system SHALL include a final written report describing how OpenSpec guided development and the delivered artifacts.

#### Scenario: Report present with specified title
- **WHEN** reviewing project documentation
- **THEN** a final report exists titled "Building a Spam Email Classification Streamlit App using OpenSpec" summarizing phases, automation, specs, implementation, visualization, and deployment

