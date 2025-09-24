from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from prophet import Prophet
import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PLOTS_FOLDER'] = 'static/plots'  # Assuming you have a folder for plots in static directory

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load the uploaded CSV file
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df[['year', 'Month']].assign(day=1))
        df['Units Sold'] = df['Units Sold'].str.replace(',', '').astype(int)

        models = df['Model'].unique()
        results = {}

        for model_name in models:
            print(f"Processing model: {model_name}")

            # Filter data for the current model
            df_model = df[df['Model'] == model_name].copy()

            # Rename columns for Prophet
            df_model = df_model[['date', 'Units Sold']].rename(columns={'date': 'ds', 'Units Sold': 'y'})

            # Train the Prophet model
            model = Prophet()
            model.fit(df_model)

            # Create future dates for prediction (2021)
            future_dates = pd.date_range(start='2021-01-01', end='2021-12-01', freq='MS')
            future_df = pd.DataFrame({'ds': future_dates})

            # Generate predictions
            prediction = model.predict(future_df)

            # Calculate MAPE and accuracy
            aligned_predictions = prediction.set_index('ds').loc[df_model['ds'].min():, 'yhat']
            mape = mean_absolute_percentage_error(df_model.set_index('ds')['y'], aligned_predictions)
            accuracy = 100 - mape

            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': prediction[['ds', 'yhat']]
            }

            # Save plots for each model
            plot_forecast(model, prediction, model_name)
            plot_components(model, prediction, model_name)

        return render_template('results.html', results=results)

    return render_template('index.html')

@app.route('/graphs/<model_name>')
def show_graphs(model_name):
    forecast_plot = url_for('static', filename=f'plots/{model_name}_forecast.png')
    components_plot = url_for('static', filename=f'plots/{model_name}_components.png')
    return render_template('graphs.html', model_name=model_name, forecast_plot=forecast_plot, components_plot=components_plot)

def plot_forecast(model, prediction, model_name):
    plt.figure()
    model.plot(prediction)
    plt.title(f'Forecast for {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plot_filename = f'{model_name}_forecast.png'
    plot_filepath = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

def plot_components(model, prediction, model_name):
    plt.figure()
    model.plot_components(prediction)
    plt.title(f'Components for {model_name}')
    plot_filename = f'{model_name}_components.png'
    plot_filepath = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
