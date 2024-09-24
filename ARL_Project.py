#########################
# Business Problem
#########################

# Armut, Turkey's largest online service platform, brings together service providers and service seekers.
# It allows easy access to services such as cleaning, repair, transportation with just a few clicks
# on a computer or smartphone.
# Using the dataset containing users and the services they have received,
# we want to build a product recommendation system using Association Rule Learning.


#########################
# Dataset
#########################
# The dataset consists of services received by users and the categories of these services.
# It contains the date and time when each service was purchased.

# UserId: Customer ID
# ServiceId: Anonymized services belonging to each category. (Example: Sofa cleaning service under the cleaning category)
# A ServiceId can be found under different categories and represents different services under different categories.
# (Example: The service with CategoryId 7 and ServiceId 4 is radiator cleaning, whereas with CategoryId 2 and ServiceId 4, it is furniture assembly)
# CategoryId: Anonymized categories. (Example: Cleaning, transportation, repair)
# CreateDate: The date when the service was purchased


#########################
# TASK 1: Data Preparation
#########################
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
# Ensures that the output fits in one line.
pd.set_option("display.expand_frame_repr", False)
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the "armut_data.csv" file.

df_ = pd.read_csv("Tavsiye Sistemleri/armut_data.csv")

df = df_.copy()

df.head(10)
df.info()
df_.shape
df.shape

df.isnull().sum()

# Step 2: ServiceID represents a different service under each CategoryID.
# Create a new variable that represents the services by concatenating ServiceID and CategoryID with an underscore ("_").

df["ServiceCategory"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head(10)

# Step 3: The dataset consists of the date and time the services were received, but there is no basket (invoice) definition.
# To apply Association Rule Learning, we need to define a basket (like an invoice).
# Here, the basket definition will be the monthly services received by each user. For example;
# The services 9_4, 46_4 received by user 7256 in August 2017 will form one basket;
# The services 9_4, 38_4 received in October 2017 will form another basket.
# We need to define these baskets with a unique ID.
# To do this, first create a new date variable that contains only year and month.
# Then, combine UserID and the newly created date variable with an underscore ("_") to create a new variable called ID.


# Create a new 'date' variable containing only year and month from the 'CreateDate' column
df["date"] = pd.to_datetime(df["CreateDate"]).dt.to_period("M")
df.head()

# Create a new 'ID' variable by concatenating 'date' and 'UserId' with an underscore
df["ID"] = df["date"].astype(str) + "_" + df["UserId"].astype(str)
df.head()

#########################
# TASK 2: Generate Association Rules
#########################

# Step 1: Create a pivot table of basket services as shown below.

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..
# Steps to pivot the basket services
basket_pivot = (df
                .groupby(['ID', 'ServiceCategory'])['ServiceCategory']
                .count().unstack().reset_index().fillna(0)
                .set_index('ID'))

# Convert the pivot table to a boolean matrix
basket_pivot = basket_pivot.apply(lambda x: x > 0).astype(int)
basket_pivot.head()

# Step 2: Generate association rules.

# Find frequent itemsets by setting a minimum support threshold
frequent_itemsets = apriori(basket_pivot, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


# Step 3: Use the arl_recommender function to recommend a service to a user who last received the 2_0 service.

def arl_recommender(rules_df, service, rec_count=1):
    """
    This function recommends a service to a user who has received a particular service.

    :param rules_df: Dataframe of association rules
    :param service: The service for which recommendations will be made (e.g., '2_0')
    :param rec_count: Number of recommendations to make
    :return: Recommended services
    """
    # Find rules where the antecedents include the specified service
    sorted_rules = rules_df[rules_df['antecedents'].apply(lambda x: service in x)].sort_values("lift", ascending=False)

    # Return the top N (rec_count) recommendations from the consequents
    return sorted_rules["consequents"].head(rec_count)


# If a user last received the 2_0 service, we will recommend the services with the highest lift values.

recommendations = arl_recommender(rules, "2_0", 3)  # Let's make 3 recommendations
recommendations