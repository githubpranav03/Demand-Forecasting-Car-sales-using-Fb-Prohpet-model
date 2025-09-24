# Demand Forecasting for Car Sales Using Prophet

This project is a machine learning model that predicts monthly car sales using Facebook's Prophet forecasting tool. It aims to provide accurate demand forecasting for various car models and brands.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [Demo Video](#demo-video)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project utilizes historical sales data to predict future sales trends for car models. The model is built using Facebook Prophet, a robust forecasting tool designed for time series data. This can be useful for car dealerships and manufacturers to plan their inventory and marketing strategies.

## Features

- Predicts monthly car sales for different car brands and models.
- Utilizes Facebook Prophet for time series forecasting.
- Provides a web interface for users to upload sales data and get predictions.
- Visualizes historical and predicted sales data.

## Installation

1. Clone the repository:
   git clone https://github.com/SKNAID/Demand-Forecasting-car-sales.git
   cd Demand-Forecasting-car-sales

2. Create and activate a virtual environment:
   
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   
4. Install the required dependencies:

   pip install -r requirements.txt
   
## Usage

1. Run the Flask app:
   
   python app.py

3. Open your browser and navigate to:
   
   http://127.0.0.1:5000/
   

5. Upload the sales data CSV file and get predictions.

## Project Structure
.
├── app.py              # Flask application file
├── env/                # Virtual environment directory
├── static/             # Static files (CSS, JS, images)
├── templates/          # HTML templates
├── uploads/            # Directory for uploaded files
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


## Dataset

The dataset contains monthly sales data from 2013 to 2021 for various car brands and models. The columns include:
- `year`: The year of the sale.
- `month`: The month of the sale.
- `make`: The car brand.
- `model`: The car model.
- `units_sold`: Number of units sold.

## Modeling

The Prophet model is trained using historical sales data. The training period is from 2013 to 2020, and the model is tested on data from 2021.

### Model Training

The model is trained separately for each car brand to improve accuracy. Adjustments to hyperparameters and seasonality settings are made to optimize performance.

### Evaluation

The model's performance is evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

## Results

The results show improved forecasting accuracy for most car brands, with a significant increase in predictive power after tuning the model's parameters.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## DEMO VIDEO
Watch a demo of the project in action here: Demo Video https://vimeo.com/991406926?share=copy

## Acknowledgements

- [Facebook Prophet](https://github.com/facebook/prophet) for the forecasting tool.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- All contributors and users for their feedback and support.
Feel free to edit and expand upon this template to better fit the specifics of your project.
