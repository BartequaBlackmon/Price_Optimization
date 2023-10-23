#!/usr/bin/env python
# coding: utf-8

# # Price Optimisation based on price elasticity of Demand

# Price elasticity of demand (Epd), or elasticity, is the degree to which the effective desire for something changes as its price changes.Elasticity is a crucial concept in economics because it helps to understand how changes in price and income affect the behavior of consumers and producers in markets.Price elasticity precisely measures the percentage change in quantity demanded resulting from a one percent increase in price, while keeping all other factors constant.

# This project focuses on analyzing the sales of items in a cafe, including burgers, coke, lemonade, and coffee. As a data scientist, our goal is to determine the optimal pricing strategy. Setting prices too high may lead to decreased sales, while setting them too low may reduce profit margins. We aim to find the ideal balance that maximizes overall profit.

# In[1]:


# install the required packages
get_ipython().system('pip install pandas==1.1.5')
get_ipython().system('pip install numpy==1.19.5')
get_ipython().system('pip install statsmodels==0.10.2')
get_ipython().system('pip install  matplotlib==3.2.2')
get_ipython().system('pip install seaborn==0.11.1')
get_ipython().system('pip install scipy')
get_ipython().system('pip install --upgrade statsmodels scipy')


# In[2]:


# Import the reqiured libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns; sns.set(style="ticks", color_codes=True)


# In[3]:


## Get multiple outputs in the same cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

## Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# In[4]:


## Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# # Load the data

# In[8]:


# Loading the data
sold = pd.read_csv(r"C:\Users\black\OneDrive\Desktop\Machine_learning_project\selldata.csv")
transactions = pd.read_csv(r"C:\Users\black\OneDrive\Desktop\Machine_learning_project\transactionstore.csv")
date_info = pd.read_csv(r"C:\Users\black\OneDrive\Desktop\Machine_learning_project\dateinfo.csv")


# In[9]:


# check for first 5 rows
sold.head()


# In[10]:


# describe 
sold.describe()
sold.describe(include = ['O']) 


# In[11]:


sold.dtypes


# In[12]:


# check for null
sold[sold.isnull().any(axis=1)]


# SELL_ID: a categorical variable, identifier of the combination of items that is contained in the product.
# 
# SELL_CATEGORY: “0” identifies single products; the category “2” identifies the combo ones.
# 
# ITEM_ID: a categorical variable, identifier of the item that is contained in the product.
# 
# ITEM_NAME: a categorical variable, identifying the name of the item

# In[13]:


# plot a pairplot for the data
sns.pairplot(sold)


# In[14]:


# check for first 5 rows
transactions.head()


# In[15]:


# describe
transactions.describe()
transactions.describe(include = ['O'])


# In[16]:


transactions.dtypes


# In[17]:


# check for nulls
transactions[transactions.isnull().any(axis=1)]


# Important: It’s supposed the PRICE for that product in that day will not vary.
# 
# In details:
# CALENDAR_DATE: a date/time variable, having the time always set to 00:00 AM.
# 
# PRICE: a numeric variable, associated with the price of the product identified by the SELL_ID.
# 
# QUANTITY: a numeric variable, associated with the quantity of the product sold, identified by the SELL_ID.
# 
# SELL_ID: a categorical variable, identifier of the product sold.
# 
# SELL_CATEGORY: a categorical variable, category of the product sold.

# In[18]:


# plot histogram to check data distribution
plt.hist(transactions.PRICE)


# In[19]:


# plot a pairplot for the data
sns.pairplot(transactions)


# In[20]:


# check for first 5 rows
date_info.head()


# In[21]:


# describe
date_info.describe()
date_info.describe(include = ['O'])


# In[22]:


# check datatypes
date_info.dtypes


# In[23]:


# check for null
date_info[date_info.isnull().any(axis=1)]


# In[24]:


# null value imputation
date_info['HOLIDAY'] = date_info['HOLIDAY'].fillna("No Holiday")


# In[25]:


date_info


# In[26]:


# pairplot 
sns.pairplot(date_info)


# # Understanding the data better

# In[27]:


# check for unique values
np.unique(date_info['HOLIDAY'])


# In[28]:


# minimum date
date_info['CALENDAR_DATE'].min()


# In[29]:


# maximum date
date_info['CALENDAR_DATE'].max()


# In[30]:


# shape of data
date_info.shape


# In[31]:


# check for null
date_info[date_info.isnull().any(axis=1)]


# 

# In[32]:


# concatenate the data
pd.concat([sold.SELL_ID, pd.get_dummies(sold.ITEM_NAME)], axis=1)


# In[33]:



pd.concat([sold.SELL_ID, pd.get_dummies(sold.ITEM_NAME)], axis=1).groupby(sold.SELL_ID).sum()


# In[34]:


# merge the data
data1 = pd.merge(sold.drop(['ITEM_ID'],axis=1), transactions.drop(['SELL_CATEGORY'], axis= 1), on =  'SELL_ID')
data1.head(20)
b = data1.groupby(['SELL_ID', 'SELL_CATEGORY', 'ITEM_NAME', 'CALENDAR_DATE','PRICE']).QUANTITY.sum()


# In[35]:


b


# In[36]:


data1.shape # check the shape
intermediate_data = b.reset_index()


# In[37]:


data1.shape # check the shape


# In[38]:


b.shape # check the shape 


# In[39]:


# first 5 rows
intermediate_data.head()


# In[40]:


# check the minimum date
intermediate_data['CALENDAR_DATE'].min()


# In[41]:


# check the maximum date
intermediate_data['CALENDAR_DATE'].max()


# In[42]:


# merge the data
combined_data = pd.merge(intermediate_data, date_info, on = 'CALENDAR_DATE')
combined_data.head()


# In[43]:


# check for the shape
combined_data.shape


# In[44]:


combined_data[combined_data.isnull().any(axis=1)]


# In[45]:


np.unique(combined_data['HOLIDAY'])
np.unique(combined_data['IS_WEEKEND'])
np.unique(combined_data['IS_SCHOOLBREAK'])


# In[46]:


bau_data = combined_data[(combined_data['HOLIDAY']=='No Holiday') & (combined_data['IS_SCHOOLBREAK']==0) & (combined_data['IS_WEEKEND']==0)]


# In[47]:


bau_data.head()


# In[48]:


bau_data.shape


# In[49]:


# check for unique
np.unique(bau_data['HOLIDAY'])
np.unique(bau_data['IS_WEEKEND'])
np.unique(bau_data['IS_SCHOOLBREAK'])


# In[50]:


bau_data[bau_data['IS_WEEKEND']==1]


# In[51]:


bau_data[bau_data['HOLIDAY']!='No Holiday']


# In[52]:


# Data exploration
plt.hist(bau_data.ITEM_NAME)


# In[53]:


# histogram plot
plt.hist(bau_data.PRICE)


# In[54]:


# scatter plot 
plt.scatter(combined_data['PRICE'], combined_data['QUANTITY'])


# In[55]:


# scatter plot 
plt.scatter(bau_data['PRICE'], bau_data['QUANTITY'])


# In[56]:


sns.pairplot(combined_data[['PRICE','QUANTITY','ITEM_NAME']], hue = 'ITEM_NAME', plot_kws={'alpha':0.1})


# In[57]:


sns.pairplot(bau_data[['PRICE','QUANTITY','ITEM_NAME']], hue = 'ITEM_NAME', plot_kws={'alpha':0.1})


# The price density plot is bimodal. From the graph we can see that for all quantities, as the price is increased the quantity sold is decreased. Although coke is hidden in this view. We can go ahead and calculate the price elasticities for this.

# In[58]:


burger = combined_data[combined_data['ITEM_NAME'] == 'BURGER']
burger.head()
burger.shape
burger.describe()
sns.scatterplot(x = burger.PRICE, y = burger.QUANTITY )


# From the above scatter plot it is clearly visible that there must be different types of burgers being sold. Now let's see the same distributin whenwe differentiate with SELL_ID which indicates if the burger was a part of the combo and hence, must be treated separately.

# In[59]:


burger = combined_data[combined_data['ITEM_NAME'] == 'BURGER']
# print(burger)
# print(burger.describe())
sns.scatterplot(data = burger, x = burger.PRICE, y = burger.QUANTITY , hue = 'SELL_ID', legend=False, alpha = 0.1)


# In[60]:


np.unique(combined_data.SELL_ID)


# In[61]:


np.unique(combined_data.SELL_CATEGORY)


# In[62]:


burger_1070 = combined_data[(combined_data['ITEM_NAME'] == 'BURGER') & (combined_data['SELL_ID'] == 1070)]

burger_1070.head()
burger_1070.describe()
sns.scatterplot(data = burger_1070, x = burger_1070.PRICE, y = burger_1070.QUANTITY, alpha = 0.1)


# # Modeling

# In[63]:


# This is for the combined data
burger_model = ols("QUANTITY ~ PRICE", data=burger_1070).fit()
print(burger_model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(burger_model, fig=fig)


# In[ ]:





# In[64]:


burger = bau_data[bau_data['ITEM_NAME'] == 'BURGER'] # for burger
burger.head()
burger.shape
burger.describe()
sns.scatterplot(x = burger.PRICE, y = burger.QUANTITY )


# In[65]:


burger = bau_data[bau_data['ITEM_NAME'] == 'BURGER']
# print(burger)
# print(burger.describe())
sns.scatterplot(data = burger, x = burger.PRICE, y = burger.QUANTITY , hue = 'SELL_ID', legend=False, alpha = 0.1)


# In[66]:


# check for unique values
np.unique(bau_data.SELL_ID)


# In[67]:


# check for unique values
np.unique(bau_data.SELL_CATEGORY)


# In[68]:


burger_1070 = bau_data[(bau_data['ITEM_NAME'] == 'BURGER') & (bau_data['SELL_ID'] == 1070)]

burger_1070.head()
burger_1070.describe()
sns.scatterplot(data = burger_1070, x = burger_1070.PRICE, y = burger_1070.QUANTITY, alpha = 0.1)


# As you can see, the scatter plot is much cleaner. Although there does seem to be 2 separate trends

# In[69]:


burger_model = ols("QUANTITY ~ PRICE", data=burger_1070).fit()
print(burger_model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(burger_model, fig=fig)


# Let's look at the bau data again to see if there is anything els ein the data we can use to further refine our model.

# In[70]:


bau_data.head()


# In[71]:


bau2_data = combined_data[(combined_data['HOLIDAY']=='No Holiday') & (combined_data['IS_SCHOOLBREAK']==0) & (combined_data['IS_WEEKEND']==0) & (combined_data['IS_OUTDOOR']==1)]


# In[72]:


burger_1070 = bau2_data[(bau2_data['ITEM_NAME'] == 'BURGER') & (bau2_data['SELL_ID'] == 1070)]

burger_1070.head()
burger_1070.describe()
sns.scatterplot(data = burger_1070, x = burger_1070.PRICE, y = burger_1070.QUANTITY, alpha = 0.1)


# In[73]:


burger_model = ols("QUANTITY ~ PRICE", data=burger_1070).fit()
print(burger_model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_ccpr(burger_model, "PRICE")


# In[74]:


# plot 
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(burger_model, "PRICE", fig=fig)


# In[75]:


burger_2051 = combined_data[(combined_data['ITEM_NAME'] == 'BURGER') & (combined_data['SELL_ID'] == 2051)]

burger_2051.head()
burger_2051.describe()
sns.scatterplot(data = burger_2051, x = burger_2051.PRICE, y = burger_2051.QUANTITY, alpha = 0.1)


# In[76]:


burger_model = ols("QUANTITY ~ PRICE", data=burger_2051).fit()
print(burger_model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(burger_model, fig=fig)


# In[77]:


coke = combined_data[combined_data['ITEM_NAME'] == 'COKE'] # for coke
coke.head()
coke.shape
coke.describe()
sns.scatterplot(x = coke.PRICE, y = coke.QUANTITY , alpha = 0.1)


# In[78]:


coke_model = ols("QUANTITY ~ PRICE", data=coke).fit() # build and fir the model
print(coke_model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(coke_model, fig=fig)
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(coke_model, 'PRICE', fig=fig)


# In[79]:


df = combined_data[combined_data['ITEM_NAME'] == 'COFFEE'] # for coffee
df.head()
df.shape
df.describe()
sns.scatterplot(x = df.PRICE, y = df.QUANTITY , alpha = 0.1)


# In[80]:


model = ols("QUANTITY ~ PRICE", data=df).fit() # build and fit the model
print(model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'PRICE', fig=fig)


# In[ ]:





# In[ ]:





# In[81]:


df = combined_data[combined_data['ITEM_NAME'] == 'LEMONADE'] # for lemonade
df.head()
df.shape
df.describe()
sns.scatterplot(x = df.PRICE, y = df.QUANTITY , alpha = 0.1)


# In[82]:


model = ols("QUANTITY ~ PRICE", data=df).fit() # build and fit the model
print(model.summary())
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'PRICE', fig=fig)


# In[ ]:





# In[83]:


elasticities = {}


# In[84]:


# function to create a model and finding elasticity 
def create_model_and_find_elasticity(data):
    model = ols("QUANTITY ~ PRICE", data).fit() # fit the model
    price_elasticity = model.params[1]
    print("Price elasticity of the product: " + str(price_elasticity))
    print(model.summary()) # check for summary 
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_partregress_grid(model, fig=fig) # plot
    return price_elasticity, model


# In[85]:


price_elasticity, model_burger_1070 = create_model_and_find_elasticity(burger_1070)
elasticities['burger_1070'] = price_elasticity


# In[86]:


burger2051_data = bau2_data[(bau2_data['ITEM_NAME'] == "BURGER") & (bau2_data['SELL_ID'] == 2051)]
elasticities['burger_2051'], model_burger_2051 = create_model_and_find_elasticity(burger2051_data)


# In[87]:


burger2052_data = bau2_data[(bau2_data['ITEM_NAME'] == "BURGER") & (bau2_data['SELL_ID'] == 2052)]
elasticities['burger_2052'], model_burger_2052 = create_model_and_find_elasticity(burger2052_data)


# In[88]:


burger2053_data = bau2_data[(bau2_data['ITEM_NAME'] == "BURGER") & (bau2_data['SELL_ID'] == 2053)]
elasticities['burger_2053'], model_burger_2053 = create_model_and_find_elasticity(burger2053_data)


# In[89]:


coke_data = bau2_data[bau2_data['ITEM_NAME'] == "COKE"]
create_model_and_find_elasticity(coke_data)


# 2 coke are available in combo, while 1 is available as single.. So it is likely that the bottom distribution belongs to single purchases of coke. Let's verfy this

# In[91]:


coke_data


# In[92]:


coke_data_2053 = bau2_data[(bau2_data['ITEM_NAME'] == "COKE") & (bau2_data['SELL_ID'] == 2053)]
elasticities['coke_2053'], model_coke_2053 = create_model_and_find_elasticity(coke_data_2053)


# In[93]:


coke_data_2051 = bau2_data[(bau2_data['ITEM_NAME'] == "COKE") & (bau2_data['SELL_ID'] == 2051)]
elasticities['coke_2051'], model_coke_2051 = create_model_and_find_elasticity(coke_data_2051)


# In[94]:


lemonade_data_2052 = bau2_data[(bau2_data['ITEM_NAME'] == "LEMONADE") & (bau2_data['SELL_ID'] == 2052)]
elasticities['lemonade_2052'], model_lemonade_2052 = create_model_and_find_elasticity(lemonade_data_2052)


# In[97]:


coffee_data_2053 = bau2_data[(bau2_data['ITEM_NAME'] == "COFFEE") & (bau2_data['SELL_ID'] == 2053)]
elasticities['coffee_2053'], model_coffee_2053 = create_model_and_find_elasticity(coffee_data_2053)


# In[ ]:


coffee_data_3055 = bau2_data[(bau2_data['ITEM_NAME'] == "COFFEE") & (bau2_data['SELL_ID'] == 3055)]
elasticities['coffee_3055'], model_coffee_3055 = create_model_and_find_elasticity(coffee_data_3055)


# ## List in a table the items and their price elasticities

# In[98]:


# check the elastcities
elasticities


# # Find optimal price for maximum profit

# Now, let's take coke (the sell_id was 2051 for the last coke data) and since we do not the buying price of coke, let''s assume it to be a little less than the minimum coke price in the dataset

# In[99]:


coke_data = coke_data_2051


# In[100]:


# minimum value
coke_data.PRICE.min()


# In[101]:


# maximum value
coke_data.PRICE.max()


# Let's take 9 as the buying price of coke. We now want to be able to set the price of coke to get the maximum profit. PRICE is the selling price

# In[102]:


buying_price_coke = 9


# $$coke data.PROFIT = (coke data.PRICE - buying price coke) * coke data.QUANTITY$$
# Let's see the profit for various price points:

# In[ ]:





# In[103]:


start_price = 9.5
end_price = 20


# In[104]:


test = pd.DataFrame(columns = ["PRICE", "QUANTITY"])


# In[105]:


test['PRICE'] = np.arange(start_price, end_price,0.01)


# In[106]:


test['QUANTITY'] = model_coke_2051.predict(test['PRICE'])


# In[107]:


test


# In[108]:


test['PROFIT'] = (test["PRICE"] - buying_price_coke) * test["QUANTITY"]


# In[109]:


test


# In[110]:


# plot the test 
plt.plot(test['PRICE'],test['QUANTITY'])
plt.plot(test['PRICE'],test['PROFIT'])
plt.show()


# Let's find the exact price at which maximum profit is gained:

# In[111]:


ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]


# In[112]:


test.loc[[ind]]


# In[ ]:





# In[113]:


# define a function for finding the optimal price
def find_optimal_price(data, model, buying_price):
    start_price = data.PRICE.min() - 1              # start price
    end_price = data.PRICE.min() + 10               # end price
    test = pd.DataFrame(columns = ["PRICE", "QUANTITY"])  # choose required columns
    test['PRICE'] = np.arange(start_price, end_price,0.01)
    test['QUANTITY'] = model.predict(test['PRICE'])         # make predictions
    test['PROFIT'] = (test["PRICE"] - buying_price) * test["QUANTITY"]
    plt.plot(test['PRICE'],test['QUANTITY'])       # plot the results 
    plt.plot(test['PRICE'],test['PROFIT']) 
    plt.show()
    ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
    values_at_max_profit = test.iloc[[ind]]
    return values_at_max_profit
    


# ## Calculate the optimal price for all and list in table

# In[114]:


optimal_price = {}
buying_price = 9


# In[115]:


optimal_price['burger_1070'] = find_optimal_price(burger_1070, model_burger_1070, buying_price)


# In[116]:


optimal_price


# In[117]:


optimal_price['burger_2051'] = find_optimal_price(burger2051_data, model_burger_2051, buying_price)


# In[118]:


optimal_price['burger_2052'] = find_optimal_price(burger2052_data, model_burger_2052, buying_price)


# In[119]:


optimal_price['burger_2053'] = find_optimal_price(burger2053_data, model_burger_2053, buying_price)


# In[120]:


optimal_price['coke_2051'] = find_optimal_price(coke_data_2051, model_coke_2051, buying_price)


# In[121]:


optimal_price['coke_2053'] = find_optimal_price(coke_data_2053, model_coke_2053, buying_price)


# In[122]:


optimal_price['lemonade_2052'] = find_optimal_price(lemonade_data_2052, model_lemonade_2052, buying_price)


# In[123]:


optimal_price['coffee_2053'] = find_optimal_price(coffee_data_2053, model_coffee_2053, buying_price)


# In[124]:


optimal_price


# In[125]:


coke_data_2051.PRICE.describe()


# # Conclusion

# This is the price the cafe should set on it's item to earn maximum profit based on it's previous sales data. It is important to note that this is on a normal day. On 'other' days such as a holiday, or an event taking place have a different impact on customer buying behaviours and pattern. Usually an increase in consumption is seen on such days. These must be treated separately. Similarly, it is important to remove any external effects other than price that will affect the purchase behaviours of customers including the datapoints when the item was on discount.

# Once, the new prices are put up, it is important to continuously monitor the sales and profit. If this method of pricing is a part of a product, a dashboard can be created for the purpose of monitoring these items and calculating the change in the profit.
