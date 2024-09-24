# ARL-based-service-recommenders

Armut Service Recommendation System

This repository contains a Service Recommendation System developed for Armut, Turkey’s largest online service platform. The project uses Association Rule Learning (ARL) to recommend services based on users’ previous service history.

Project Overview

Armut connects service providers with users seeking services such as cleaning, repair, transportation, and more. Using this system, we aim to generate recommendations by analyzing the services customers have used in the past and finding associations between them.

Dataset

The dataset contains records of the services received by users. Each record includes:

	•	UserId: The unique identifier for each user.
	•	ServiceId: The anonymized ID of the service received. A ServiceId can represent different services under different categories.
	•	CategoryId: The anonymized ID of the category that the service belongs to.
	•	CreateDate: The date and time when the service was provided.

Objective

The goal of this project is to create a product recommendation system using Association Rule Learning (ARL). Specifically, we want to recommend relevant services to users based on the services they’ve previously purchased.

Workflow

1. Data Preparation

	•	Combined ServiceId and CategoryId into a new feature ServiceCategory to represent unique services.
	•	Created a new feature ID to represent each user’s monthly service purchases. This was used to simulate a “basket” for each user.

2. Pivot Table

Generated a pivot table where each row represents a user’s monthly service purchases, and each column corresponds to a unique service category.

3. Association Rules

Used Apriori algorithm to generate frequent itemsets and Association Rule Learning to find relationships between services. The system generates recommendations based on the lift metric.

4. Recommendations

A function (arl_recommender) is implemented to recommend services to users based on their most recent service usage. For example, if a user last received the service 2_0, the system recommends other services that are commonly purchased with 2_0.


How to Use

	1.	Install Dependencies
You need to install the following Python libraries:

pip install pandas mlxtend

2.	Run the Code
Load the dataset and run the script. The code processes the data, generates association rules, and outputs recommendations.
	3.	Example
To recommend services for a user who last used service 2_0, use the arl_recommender function like this:
recommendations = arl_recommender(rules, "2_0", 3)
print(recommendations)

Files

	•	armut_data.csv: The dataset used for this project.
	•	service_recommendation.py: The Python script that implements the recommendation system.
