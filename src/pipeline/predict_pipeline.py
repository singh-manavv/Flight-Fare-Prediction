from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime


def preprocess_form_data(form_data):
    # Convert form data to datetime objects
    date_dep = pd.to_datetime(form_data["Dep_Time"], format="%Y-%m-%dT%H:%M")
    date_arr = pd.to_datetime(form_data["Arrival_Time"], format="%Y-%m-%dT%H:%M")
    
    # Calculate duration
    duration = date_arr - date_dep
    if duration.days < 0:
        duration += pd.Timedelta(days=1)
    duration_in_minutes = duration.seconds // 60
    
    # Encode categorical variables
    airlines = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
                'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
                'Trujet','Vistara', 'Vistara Premium economy']
    sources = ['Chennai','Delhi', 'Kolkata', 'Mumbai' ]
    destinations = ['Cochin', 'Delhi','Hyderabad','Kolkata', 'New Delhi']
    
    # Initialize encoded features with 0
    encoded_features = {f'Airline_{airline}': 0 for airline in airlines}
    encoded_features.update({f'Source_{source}': 0 for source in sources})
    encoded_features.update({f'Destination_{destination}': 0 for destination in destinations})
    
    # Set the appropriate feature to 1 based on form data
    encoded_features[f'Airline_{form_data["airline"]}'] = 1
    encoded_features[f'Source_{form_data["Source"]}'] = 1
    encoded_features[f'Destination_{form_data["Destination"]}'] = 1
    
    # Prepare the final feature vector
    df = pd.DataFrame([[
        duration_in_minutes,
        int(form_data["Total_Stops"]),
        date_dep.month,
        date_dep.day
    ] + list(encoded_features.values())], columns=[
        'Duration',
        'Total_Stops',
        'Month',
        'Day'  
    ] + list(encoded_features.keys()))
    
    return df