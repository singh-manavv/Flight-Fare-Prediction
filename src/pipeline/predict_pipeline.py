import pandas as pd
from dataclasses import dataclass

@dataclass
class FlightDataPreprocessor:
    def __init__(self):
        self.airlines = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
                        'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
                        'Trujet', 'Vistara', 'Vistara Premium economy']
        self.sources = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
        self.destinations = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']

    def convert_to_datetime(self, datetime_str):
        return pd.to_datetime(datetime_str, format="%Y-%m-%dT%H:%M")

    def calculate_duration(self, departure_time, arrival_time):
        duration = arrival_time - departure_time
        if duration.days < 0:
            duration += pd.Timedelta(days=1)
        return duration.seconds // 60

    def encode_categorical_variables(self, form_data):
        encoded_features = {f'Airline_{airline}': 0 for airline in self.airlines}
        encoded_features.update({f'Source_{source}': 0 for source in self.sources})
        encoded_features.update({f'Destination_{destination}': 0 for destination in self.destinations})

        encoded_features[f'Airline_{form_data["airline"]}'] = 1
        encoded_features[f'Source_{form_data["Source"]}'] = 1
        encoded_features[f'Destination_{form_data["Destination"]}'] = 1

        return encoded_features

    def preprocess_form_data(self, form_data):
        departure_time = self.convert_to_datetime(form_data["Dep_Time"])
        arrival_time = self.convert_to_datetime(form_data["Arrival_Time"])

        duration_in_minutes = self.calculate_duration(departure_time, arrival_time)

        encoded_features = self.encode_categorical_variables(form_data)

        df = pd.DataFrame([[
            duration_in_minutes,
            int(form_data["Total_Stops"]),
            departure_time.month,
            departure_time.day
        ] + list(encoded_features.values())], columns=[
            'Duration',
            'Total_Stops',
            'Month',
            'Day'
        ] + list(encoded_features.keys())
        )

        return df