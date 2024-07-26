# Customer Satisfaction prediction with MLOps

## Overview
CustSatMLOps is a project aimed at predicting customer satisfaction using machine learning algorithms. It integrates MLOps practices using ZenML and MLflow for streamlined deployment and management.

## Features
- **Data Processing:** Includes scripts for data cleaning and preprocessing.
- **Model Training:** Implements various ML algorithms for training models.
- **MLOps Integration:** Utilizes ZenML for pipeline orchestration and MLflow for experiment tracking and model management.
- **Deployment:** Provides tools for deploying models into production.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mirsaidl/CustSatMLOps.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CustSatMLOps
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you
to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard), but first you must install the optional dependencies for the ZenML server:

```bash
pip install zenml["server"]
zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## Usage
1. **Run the pipeline:**
   ```bash
   python run_pipeline.py
   ```
2. **Deploy the model:**
   ```bash
   python run_deployment.py --config deploy
   python run_deployment.py --config predict
   ```
3. **Run Streamlit on local:**
   ```bash
   streamlit run app.py
   ```

## Repository Structure
- `data/`: Contains the dataset.
- `pipelines/`: Defines the ML pipelines.
- `steps/`: Contains steps for data processing, training, and evaluation.
- `src/`: Source code for the project.
- `requirements.txt`: List of dependencies.

## Acknowledgements
This project utilizes ZenML and MLflow for implementing MLOps.
