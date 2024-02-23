# Flight Fare Prediction

## Project Description

The Flight Fare Prediction project is a machine learning-based application designed to predict the fares of airline flights based on various factors such as departure date, time, route, stops, and additional services. This project aims to provide users with an estimate of flight prices, helping them to make more informed decisions when booking flights. By analyzing historical flight data, the model can predict future flight prices with a reasonable degree of accuracy.

## Data Gathering

The dataset for this project was sourced from Kaggle. It can be accessed and downloaded from the following link: [Flight Fare Prediction Dataset](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh).

## Dataset Information

The dataset used for this project comprises historical flight data, including features like:

- **Airline**: The name of the airline.
- **Date of Journey**: The date of the flight.
- **Source**: The starting point of the journey.
- **Destination**: The endpoint of the journey.
- **Route**: The route taken by the flight.
- **Dep_Time**: The time when the flight departs.
- **Arrival_Time**: The time when the flight arrives.
- **Duration**: Total duration of the flight.
- **Total_Stops**: Total stops between the source and destination.
- **Additional_Info**: Additional information about the flight.
- **Price**: The price of the flight ticket.

This data is split into training and testing sets to train and evaluate the model's performance.

## How This Project Works

The Flight Fare Prediction project works by following these steps:

- **Data Preprocessing**: Cleansing and preparing the data for training, including handling missing values, encoding categorical variables, and normalizing the data.
- **Feature Selection**: Selecting the most relevant features that influence flight prices.
- **Model Training**: Using the processed data to train a machine learning model. Various algorithms like Random Forest, Gradient Boosting, and Linear Regression can be explored to find the best performer.
- **Model Evaluation**: Evaluating the model's performance using metrics like MAE (Mean Absolute Error) and R² score.
- **Prediction**: Using the trained model to predict flight fares based on user input.

## How to Install and Run

To set up and run the Flight Fare Prediction project, follow these steps:

### Prerequisites

Ensure you have Python (version 3.11 or later) installed on your system. You will also need pip for installing Python packages.

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/Flight-Fare-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Flight-Fare-Prediction
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. To start the application, run the following command in the terminal:

   ```bash
   python app.py
   ```
2. Access the web interface by navigating to `http://localhost:5000` in your web browser.
3. Enter the required flight details and click on the predict button to view the predicted flight fare.

## Conclusion

The Flight Fare Prediction project leverages machine learning to provide users with flight fare estimates, making it easier to plan and budget for travel. By continuously updating the model with new data, its predictions can become more accurate over time, further aiding in the decision-making process for travelers.

## Preview
[![Imgur Gif]](https://imgur.com/a/9gBUjj3)]
## License

[MIT](https://choosealicense.com/licenses/mit/)
