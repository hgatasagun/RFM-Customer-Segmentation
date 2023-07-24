##############################################################
# FLO Customer Segmentation using the RFM Method
##############################################################

##############################################################
# 1. Business Problem
##############################################################
# FLO, an online shoe store, wants to segment its customers and determine
# marketing strategies based on these segments.

# Dataset story
# The dataset consists of information from customers who made their last purchases
# as OmniChannel (both online and offline) between the years 2020 and 2021.
#
# Variables
# master_id -- Unique customer number
# order_channel -- The channel used for shopping (Android, iOS, Desktop, Mobile)
# last_order_channel -- The channel used for the last purchase
# first_order_date -- The date of the customer's first purchase
# last_order_date -- The date of the customer's last purchase
# last_order_date_online -- The date of the customer's last online purchase
# last_order_date_offline -- The date of the customer's last offline purchase
# order_num_total_ever_online -- Total number of purchases made by the customer online
# order_num_total_ever_offline -- Total number of purchases made by the customer offline
# customer_value_total_ever_offline -- Total amount paid by the customer for offline purchases
# customer_value_total_ever_online -- Total amount paid by the customer for online purchases


###############################################################
# 2. Data Preparation
###############################################################

# Importing libraries
##############################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv('flo_data_20k.csv')
df = df_.copy()


# Data Understanding
#############################
def check_df(dataframe, head=10):
    print('################# Shape ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df)


# Creating new variables for each customer's total number of purchases and total spending
##########################################################################################
df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.head()


# Converting the variables representing dates to the 'date' data type
#####################################################################
date_columns = [col for col in df.columns if 'date' in col]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


# Analyzing the distribution of the number of customers in each shopping channel,
# the total number of products purchased, and the total spending
#################################################################################
df['order_channel'].hist()
plt.show()

df['order_num_total'].hist(bins=100)
plt.xlim(0, 35)
plt.show()

df['customer_value_total'].hist(bins=250)
plt.xlim(0, 5000)
plt.show()


# Ranking the top 10 customers who have brought the highest revenue
###################################################################
top10_value = df.sort_values(by='customer_value_total', ascending=False).head(10)
top10_value[['master_id', 'customer_value_total']]


# Ranking the top 10 customers who have placed the most orders
##############################################################
top10_order = df.sort_values(by='order_num_total', ascending=False).head(10)
top10_order[['master_id', 'order_num_total']]


###############################################################
# 3. Calculation of RFM Metrics
###############################################################

# Determining the analysis date as 2 days after the last order date
###################################################################
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)


# Assigning the calculated metrics to a new variable called "RFM"
#################################################################
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                   'order_num_total': lambda order_num_total: order_num_total,
                                   'customer_value_total': lambda customer_value_total: customer_value_total})
rfm.head()


# Changing the variable names to "Recency," "Frequency," and "Monetary"
#######################################################################
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()


###############################################################
# 4. Calculation of RF Score
###############################################################

# Converting the Recency, Frequency, and Monetary metrics into scores ranging from 1 to 5
##########################################################################################
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]) # (method='first') --> order-based ranking
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])


# Assigning "recency_score" and "frequency_score" as a single variable named "RF_SCORE"
#######################################################################################
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))


###############################################################
# 5. Defining RF Score as Segments
###############################################################

# Defining segments for the created RF scores
##############################################
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


# Converting scores into segments using 'seg_map'
#################################################
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


# Examining the mean values of recency, frequency, and monetary values for each segment
#######################################################################################
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])


# FLO is introducing a new women's shoe brand with products priced above the general customer
# preferences. Therefore, they want to reach out to specific customer profiles for the promotion
# and sales of this new brand. They aim to contact loyal customers (champions, loyal_customers)
# and those who have made purchases in the women's category. The goal is to identify the customer IDs
# of these specific customers and save them to a CSV file.
#######################################################################################################
df.index = df['master_id'] # assigning the 'master_id' column as the new index for 'df'
df.head()

new_df = pd.DataFrame({'segment': rfm['segment'], 'interested_in_categories_12': df['interested_in_categories_12']})
new_df.head()

flo_kadin = new_df.loc[(new_df["segment"].isin(["champions", 'loyal_customers'])) &
                       (new_df["interested_in_categories_12"].str.contains("KADIN"))] # KADIN --> woman

df_id_kadin = pd.DataFrame()
df_id_kadin["customer_id"] = flo_kadin.index
df_id_kadin.to_csv("customer_id.csv")


# A discount of nearly 40% is planned for men's and children's products.
# Customers who are interested in these discounted categories and have been good customers
# in the past but haven't made purchases for a long time, newly acquired customers,
# and those who are considered "at risk" of being lost should be specifically targeted.
# The goal is to save the IDs of the eligible customers into a CSV file for targeted marketing.
################################################################################################
flo_mc = new_df.loc[(new_df["segment"].isin(["cant_loose", "about_to_sleep", 'new_customers'])) &
                    (new_df["interested_in_categories_12"].str.contains('COCUK', 'ERKEK'))] # COCUK -> children, ERKEK -> man

df_id_mc = pd.DataFrame()
df_id_mc["customer_id2"] = flo_mc.index
df_id_mc.to_csv("customer_id2.csv")