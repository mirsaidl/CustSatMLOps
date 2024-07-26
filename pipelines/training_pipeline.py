from zenml import pipeline
from steps.ingest import ingest_data
from steps.clean import clean_data
from steps.train import train_model
from steps.evaluate import evaluate_model


@pipeline
def training_pipeline(data_path: str):
    # Define the steps
    # Step 1: Load data
    # Step 2: Preprocess data
    # Step 3: Train model
    # Step 4: Evaluate model
    # Step 5: Save model
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, y_train)
    accuracy, precision, f1 = evaluate_model(model, x_test, y_test)
    
    