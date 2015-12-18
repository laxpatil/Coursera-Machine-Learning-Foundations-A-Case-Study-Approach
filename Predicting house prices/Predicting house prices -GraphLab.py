
# coding: utf-8

# In[188]:

import graphlab as gl


# In[189]:

sales = graphlab.SFrame('home_data.gl/')


# In[190]:

sales


# In[ ]:




# In[191]:

graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")


# In[192]:

#create a simple model


# In[193]:

train_data,test_data=sales.random_split(0.8,seed=0)


# In[194]:

len(train_data)


# In[143]:

#build a model


# In[195]:

sqft_model=gl.linear_regression.create(train_data,target='price',features=['sqft_living'],validation_set=None)


# In[145]:

#evaluate model


# In[196]:

print test_data['price'].mean()


# In[197]:

sqft_model.evaluate(test_data)


# In[198]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[199]:

plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')


# In[200]:

sqft_model.get('coefficients')


# In[201]:

#other feattures


# In[202]:

my_features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']


# In[203]:

sales[my_features].show()


# In[205]:

sales.show(view='BoxWhisker Plot', x='zipcode',y='price')


# In[206]:

#regression model with more features


# In[207]:

my_features_model=gl.linear_regression.create(train_data,target='price'
                                             , features=my_features,validation_set=None)


# In[208]:

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)


# In[209]:

#apply model to predict houses printce


# In[210]:

house=sales[sales['id']=='5309101200']


# In[211]:

house


# In[212]:

house['price']


# In[213]:

print sqft_model.predict(house)


# In[214]:

print my_features_model.predict(house)


# In[215]:

house2=sales['id'=='1925069083']


# In[216]:

house2


# In[217]:

print sqft_model.predict(house2)


# In[218]:

print my_features_model.predict(house2)


# In[219]:

bill_house={'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}


# In[220]:

print sqft_model.predict(gl.SFrame(bill_house))


# In[223]:

print my_features_model.predict(gl.SFrame(bill_house))


# In[224]:

##TO ANSWER QUIZ


# In[ ]:




# In[225]:

houses_98039=sales[sales['zipcode']=='98039']


# In[226]:

#question 1
houses_98039['price'].mean()


# In[227]:

houses_2kTo4k=sales[(sales['sqft_living']>2000) & (sales['sqft_living'] <= 4000)]


# In[228]:

houses_2kTo4k


# In[229]:

#question 2

print float(len(houses_2kTo4k))/len(sales)


# In[230]:

#question 3
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[231]:

advanced_model=gl.linear_regression.create(train_data,target='price',
                                          features=advanced_features,validation_set=None)


# In[234]:

#question3
my_features_rmse=my_features_model.evaluate(test_data)
print my_features_rmse
advanced_features_rmse=advanced_model.evaluate(test_data)
print advanced_features_rmse


# In[235]:

#Question 3

print my_features_rmse['rmse'] - advanced_features_rmse['rmse']


# In[ ]:




# In[ ]:




# In[ ]:



