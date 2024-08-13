

```markdown
# House Price Prediction App

This project integrates two machine learning models—Gaussian Process Regressor (GPR) and Prophet—for predicting house prices. It features a FastAPI backend for model predictions and a Streamlit frontend for user interaction. The GPR model predicts house prices based on various input features, while the Prophet model forecasts future prices based on historical data.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies](#technologies)
- [Setup and Installation](#setup-and-installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#usage)
  - [Gaussian Process Regressor (GPR) Prediction](#gaussian-process-regressor-gpr-prediction)
  - [Prophet Forecast](#prophet-forecast)
- [Endpoints](#endpoints)
  - [GPR Prediction](#gpr-prediction)
  - [Prophet Forecast](#prophet-forecast-1)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project consists of two main components:
1. **FastAPI Backend**: Provides RESTful APIs to interact with the machine learning models.
2. **Streamlit Frontend**: A web application for users to input features and view predictions from both models.

### Features
- Predict house prices using the Gaussian Process Regressor model.
- Forecast future house prices using the Prophet model.
- Visualize predictions with graphs for better interpretation.

## Technologies

- **FastAPI**: A modern web framework for building APIs with Python 3.7+.
- **Streamlit**: A framework for building interactive web apps for machine learning and data science.
- **Prophet**: A forecasting tool by Facebook for time series data.
- **Scikit-learn**: A machine learning library for Python.
- **Matplotlib**: A plotting library for Python.

## Setup and Installation

### Backend Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Backend Dependencies**

   ```bash
   pip install fastapi uvicorn pandas scikit-learn prophet joblib
   ```

4. **Download the Data**

   Ensure the dataset `kc_house_data.csv` is in the root directory of the project. You can obtain it from the [Kaggle House Prices dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).

5. **Run the FastAPI Server**

   ```bash
   uvicorn main:app --reload
   ```

   The FastAPI server will be available at `http://127.0.0.1:8000`.

### Frontend Setup

1. **Install Streamlit**

   ```bash
   pip install streamlit matplotlib requests
   ```

2. **Run the Streamlit Application**

   ```bash
   streamlit run app.py
   ```

   The Streamlit application will be available at `http://localhost:8501`.

## Usage

### Gaussian Process Regressor (GPR) Prediction

1. **Select the GPR model** from the sidebar in the Streamlit app.
2. **Input the house features** (e.g., number of bedrooms, square footage) into the form.
3. **Click the "Predict Price" button** to see the predicted house price.
4. **View the bar chart** displaying the predicted house price.

### Prophet Forecast

1. **Select the Prophet model** from the sidebar in the Streamlit app.
2. **Input the number of days** into the future you want to forecast.
3. **Click the "Predict Future Prices" button** to see the future price forecasts.
4. **View the line plot** with confidence intervals showing the predicted future house prices.

## Endpoints

### GPR Prediction

- **URL**: `/predict/gpr/`
- **Method**: POST
- **Request Body**: JSON with the following fields:

   ```json
   {
     "bedrooms": 3,
     "bathrooms": 2.0,
     "sqft_living": 2000,
     "sqft_lot": 5000,
     "floors": 1.0,
     "waterfront": 0,
     "view": 1,
     "condition": 3,
     "grade": 7,
     "sqft_above": 1500,
     "sqft_basement": 0,
     "yr_built": 2000,
     "yr_renovated": 2000,
     "zipcode": 98101,
     "lat": 47.0,
     "long": -122.0,
     "sqft_living15": 2000,
     "sqft_lot15": 5000,
     "year": 2020,
     "month": 8,
     "day": 15
   }
   ```

- **Response**: JSON with the predicted price:

   ```json
   {
     "predicted_price": 500000.00
   }
   ```

### Prophet Forecast

- **URL**: `/predict/prophet/`
- **Method**: POST
- **Request Body**: JSON with the following field:

   ```json
   {
     "periods": 30
   }
   ```

- **Response**: JSON with forecast data including `ds` (date), `yhat` (predicted price), `yhat_lower` (lower bound), and `yhat_upper` (upper bound):

   ```json
   [
     {"ds": "2024-09-01", "yhat": 510000.00, "yhat_lower": 500000.00, "yhat_upper": 520000.00},
     ...
   ]
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Replace `your-username` in the clone URL with your GitHub username and adjust the dataset source if needed. Make sure to test the setup and usage instructions for accuracy before sharing the repository.
```

This `README.md` file provides comprehensive instructions and information about the project, including setup, usage, and API details. Make sure to replace placeholders with actual data and check for accuracy in your environment.