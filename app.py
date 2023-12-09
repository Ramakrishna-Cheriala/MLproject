import pickle
import sys
import os

# from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


gender = {1: "Male", 2: "Female"}
print("\n\nSelect your Gender: \n", gender, "\n")
gender_n = int(input("Enter a number form the above list: "))
selected_gender = (gender[gender_n]).lower()

race_ethnicity = {1: "group A", 2: "group B", 3: "group C", 4: "group D", 5: "group E"}
print("\n\nEnter you ethnicity: \n", race_ethnicity, "\n")
race_n = int(input("Enter a number form the above list: "))
selected_race_ethnicity = race_ethnicity[race_n]

parental_level_of_education = {
    1: "associate's degree",
    2: "bachelor's degree",
    3: "high school",
    4: "master's degree",
    5: "some college",
    6: "some high school",
}
print("\n\nParental Level of Education: \n", parental_level_of_education, "\n")
parent_n = int(input("Enter a number form the above list: "))
selected_parental_level_of_education = parental_level_of_education[parent_n]

lunch = {1: "free/reduced", 2: "standard"}
print("\n\nLunch Type: \n", lunch, "\n")
lunch_n = int(input("Enter a number form the above list: "))
selected_lunch = lunch[lunch_n]

test_preparation_course = {1: "None", 2: "Completed"}
print("\n\nTest preparation Course: \n", test_preparation_course, "\n")
test_n = int(input("Enter a number form the above list: "))
selected_test_preparation_course = (test_preparation_course[test_n]).lower()


reading_score = int(input("\nEnter Reading Score: "))

writing_score = int(input("\nEnter Writing Score: "))

data = CustomData(
    gender=selected_gender,
    race_ethnicity=selected_race_ethnicity,
    parental_level_of_education=selected_parental_level_of_education,
    lunch=selected_lunch,
    test_preparation_course=selected_test_preparation_course,
    reading_score=reading_score,
    writing_score=writing_score,
)

pred_df = data.get_data_as_data_frame()
print("\n", pred_df)
predict_pipeline = PredictPipeline()
results = predict_pipeline.predict(pred_df)
print("\n\nYour Predicted maths score: ", results[0])
