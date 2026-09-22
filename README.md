# House Price Prediction 

This project converts the regression model from the supplied Lab 1 notebook into a Streamlit application.

## Files

- `app.py` - Streamlit application. It trains the Multiple Linear Regression model when the app starts, so NO `.pkl` file is required.
- `boston_housing.csv` - REQUIRED dataset. Put your existing CSV in this folder.
- `requirements.txt` - Python packages required.
- `Alicia_2648525_Lab1.ipynb` - Your original lab notebook.

## Run in VS Code

1. Open this folder in VS Code.
2. Make sure `boston_housing.csv` is in the same folder as `app.py`.
3. Open the VS Code terminal.
4. Create a virtual environment (recommended):

   Windows:
   `python -m venv .venv`

5. Activate it:

   PowerShell:
   `.venv\Scripts\Activate.ps1`

   Command Prompt:
   `.venv\Scripts\activate`

6. Install packages:

   `pip install -r requirements.txt`

7. Start Streamlit:

   `streamlit run app.py`

8. Streamlit will open the application in your browser.

##Important

The CSV must contain a column named `MEDV`. The model uses every other column as a predictor, matching the original notebook.

No model `.pkl` file is included because the Streamlit app trains the Linear Regression model directly from the CSV when it starts.
