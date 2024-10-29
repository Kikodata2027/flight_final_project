# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:38:41 2024

@author: ahmad
"""

# section 1  importing the pandas and numpy libraries

import pandas as pd 
import numpy as np 
import math  

# section 2 reading dataframe

df= pd.read_csv("D:/flights.csv")

########### section 3 : general information about the source dataframe

print(df.info())

# RangeIndex: 1048575 entries, 0 to 1048574
# Data columns (total 31 columns):
    
print(df.columns)

"""
Index(['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER',
       'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'],

"""

# understanding where is the missing values without removing or filling any missing values  in the dataframe
missing_values_series = df.isnull().sum()
missing_values_list=list(missing_values_series)
print(missing_values_series)


"""

YEAR                         0
MONTH                        0
DAY                          0
DAY_OF_WEEK                  0
AIRLINE                      0
FLIGHT_NUMBER                0
TAIL_NUMBER               7750
ORIGIN_AIRPORT               0
DESTINATION_AIRPORT          0
SCHEDULED_DEPARTURE          0
DEPARTURE_TIME           39515
DEPARTURE_DELAY          39515
TAXI_OUT                 40229
WHEELS_OFF               40229
SCHEDULED_TIME               2
ELAPSED_TIME             43071
AIR_TIME                 43071
DISTANCE                     0
WHEELS_ON                41296
TAXI_IN                  41296
SCHEDULED_ARRIVAL            0
ARRIVAL_TIME             41296
ARRIVAL_DELAY            43071
DIVERTED                     0
CANCELLED                    0
CANCELLATION_REASON    1008048
AIR_SYSTEM_DELAY        820047
SECURITY_DELAY          820047
AIRLINE_DELAY           820047
LATE_AIRCRAFT_DELAY     820047
WEATHER_DELAY           820047
dtype: int64



"""

# removing duplicates values in the main dataframe
num_duplicates = df.duplicated().sum()
df_after_drop = df.drop_duplicates()

"""
there is no duplicated values in the dataframe so the main dataframe 

"""

################ section 4 adding the delayreason colomun to the data frame depending on the previous data

column_to_check1 = 'AIR_SYSTEM_DELAY'
column_to_check2 = 'SECURITY_DELAY'
column_to_check3 = 'AIRLINE_DELAY'
column_to_check4 = 'LATE_AIRCRAFT_DELAY'
column_to_check5 = 'WEATHER_DELAY'
column_to_check6 = 'CANCELLATION_REASON'
column_to_check7 = 'ARRIVAL_DELAY'

### section 4 adding the new coloumn named as reasondelay
def check_value(row):
    
    li=[]
    li.append(row[column_to_check1])
    li.append(row[column_to_check2])
    li.append(row[column_to_check3])
    li.append(row[column_to_check4])
    li.append(row[column_to_check5])
   
    max =0
    index_max=0
    
    if (pd.isnull(df.at[1, column_to_check6])and row["ARRIVAL_DELAY"]<0) :
        index_max= 1000
         
    if (pd.isnull(df.at[1, column_to_check6])and row["ARRIVAL_DELAY"]>=0) :
        index_max = 900
    
    if  row["CANCELLATION_REASON"] =="A" or row["CANCELLATION_REASON"] =="B" or row["CANCELLATION_REASON"] =="C" or row["CANCELLATION_REASON"] =="D" :
        index_max = 800
    
        
    
    
    for i in range(len(li)):
       
        if li[i]>max :
            
            max= li[i]
            index_max= i
            
    if index_max==0 :
        return "AIRS"
    
    elif index_max == 1 :
        return "SEC"
    
    elif index_max ==2 :
        return "AIRL"
    
    elif index_max ==3 :
        return "AIRCRAFT"
    
    elif index_max ==4 :
        return "WEATHER"
    
    elif index_max ==1000 :
        return "EXCELLENT"
    
    elif index_max == 900 :
        return "perfect"
    elif index_max ==800 :
        return "cancelled"
        
 
     
    
    
df['delayreason'] = df.apply(check_value, axis=1)

# general information about the dataframe after adding the y colomn

print(df.info())

"""
RangeIndex: 1048575 entries, 0 to 1048574
Data columns (total 32 columns):
    
    the only change here is that a new colomn was added and this is why we have 32 colomns comparing to 31 colomns


"""

################## section 5 Independent anaysis ( the count of each delayreason and the percentage of count for every single reason of delay)


count_by_reason_of_delay_series = df.delayreason.value_counts()
count_by_reason_of_delay_list = list(count_by_reason_of_delay_series)
perecentage_by_reason_of_delay_list= []

for i in range(len(count_by_reason_of_delay_list)) :
    
    perecentage_by_reason_of_delay_list.append(count_by_reason_of_delay_list[i]/1048574)
    

"""

delayreason      count            %

very perfect     558964           53%

perect           218012           20%

AIRCRAFT_LATE    87438            8%

AIR_SYSTEM       69360            6%

AIRLINE          65052            6%

cancelled        40527            3%

WEATHER           8931            <1%

SECURITY         291              < 1%




"""

# visulization of count by reason of delay represented as count_by_reason_of_delay

import matplotlib.pyplot as plt
import seaborn as sns
delay_reason_series = df["delayreason"]

delay_reason_list =list(delay_reason_series)
plt.hist(delay_reason_list ,color="red")

plt.grid()
plt.show()

"""

it is obvious that around half of the entire flights arrive earlier than excepted 
while flights that had low level of delay was one-fifth of the whole flights

the data above shows that few number of flights delayed due to reasons such as weather and secuirty


"""


##### section 6 : Independent analysis (count by distance)  plot named as count_by_distance_independent analysis

sns.set(style="darkgrid")

fig,axs =plt.subplots(2,2,figsize=(10,8))

sns.histplot(data=df,x="DISTANCE",ax=axs[0,0], color="green")
sns.histplot(data=df,x="DISTANCE",ax=axs[0,1], color="skyblue")

"""

generally we can say from the count_by_distance_independent _analysis plot that the count of  most flights is smaller than 2500 

and this is why we have to study the count of distance by applying a filter condition for the distance

"""
filtered_df_distance1 = df[df["DISTANCE"]<2500]


count_by_reason_of_delay_special_case = filtered_df_distance1.delayreason.value_counts()
count_by_reason_of_delay_special_case_list =list(filtered_df_distance1.delayreason.value_counts())
perecentage_by_reason_of_delay_special_case_list= []

for i in range(len(count_by_reason_of_delay_list)) :
    
    perecentage_by_reason_of_delay_special_case_list.append(count_by_reason_of_delay_special_case_list[i]/1028145)
    

"""
 0.5325114648225688
0.2082293839876671
0.08411556735674443
0.06579713950853236
0.06175490811121
0.0388398523554557
0.008477403479081259
0.0002742803787403528

it is obvious from the data here is that the percentage of reason of delay is not effected by distance specially because we apply a filter on distance




"""


##### Independent analysis (count of flights depending on month)


plt.figure(figsize=(8, 6))
sns.countplot(x='MONTH', data=df, palette="viridis")

# Adding title and labels
plt.title('Month vs counts')
plt.xlabel('MONTH')
plt.ylabel('Count')

# Display the plot
plt.show()

"""

the number of flights in month 1 > month 2 > month 3


there is a huge difference between the number of flights in (1,2) and the month 3


"""

filtered_by_month3 = df[df["MONTH"]==2]
count_by_month3_series_cancceled=filtered_by_month3.delayreason.value_counts()





plt.figure(figsize=(8, 6))
sns.histplot(data=df, x="delayreason", hue="MONTH", multiple="stack", shrink=0.8)

# Adding title and labels
plt.title('Stacked Count Plot of DELAY REASON AND MONTH')
plt.xlabel('Category')
plt.ylabel('Count')

# Display the plot
plt.show()

"""

* month 2 has more cancellation than month 3 and month 1 

* the number of flights that arrived earlier in month 3 is more than those arrived with an acceptable range of delay(perfect)
*  in month 3 the most flights delayed due to reasons such as aircraft , airsystem and airline 

"""


############# 

#### relation between months and amount of distance


sns.boxplot(x="MONTH",y="DISTANCE",data=df)

sns.boxplot(x="delayreason",y="DISTANCE",data=df)

delay_by_reason = df.groupby('delayreason')['ARRIVAL_DELAY'].sum().reset_index()



plt.figure(figsize=(10, 6))
sns.barplot(data=delay_by_reason, x='delayreason', y='ARRIVAL_DELAY', palette="viridis")
plt.title('dep Delay by Reason of Delay')
plt.xlabel('Reason of Delay')
plt.ylabel('depart Delay hours)')
plt.show()






##############################    

#### froping unusefull coloumns from the dataframe 


"""
YEAR                         0
MONTH                        0
DAY                          0
DAY_OF_WEEK                  0
AIRLINE                      0
FLIGHT_NUMBER                0
TAIL_NUMBER               7750
ORIGIN_AIRPORT               0
DESTINATION_AIRPORT          0
SCHEDULED_DEPARTURE          0
DEPARTURE_TIME           39515
DEPARTURE_DELAY          39515
TAXI_OUT                 40229
WHEELS_OFF               40229
SCHEDULED_TIME               2
ELAPSED_TIME             43071
AIR_TIME                 43071
DISTANCE                     0
WHEELS_ON                41296
TAXI_IN                  41296
SCHEDULED_ARRIVAL            0
ARRIVAL_TIME             41296
ARRIVAL_DELAY            43071
DIVERTED                     0
CANCELLED                    0
CANCELLATION_REASON    1008048
AIR_SYSTEM_DELAY        820047
SECURITY_DELAY          820047
AIRLINE_DELAY           820047
LATE_AIRCRAFT_DELAY     820047
WEATHER_DELAY           820047
"""

"""
* we are going to drop 
 AIRLINE                      0
FLIGHT_NUMBER                0
TAIL_NUMBER               7750
ORIGIN_AIRPORT               0
DESTINATION_AIRPORT          0
SCHEDULED_DEPARTURE          0
DEPARTURE_TIME           39515

TAXI_OUT                 40229
WHEELS_OFF               40229
SCHEDULED_TIME               2
ELAPSED_TIME             43071
AIR_TIME                 43071

WHEELS_ON                41296
TAXI_IN                  41296
SCHEDULED_ARRIVAL            0
ARRIVAL_TIME             41296

DIVERTED                     0
CANCELLED                    0
CANCELLATION_REASON    1008048
"""



df_after_drop = df.drop(columns=['AIRLINE'])

df_after_drop = df_after_drop.drop(columns=['FLIGHT_NUMBER'])
df_after_drop = df_after_drop.drop(columns=['TAIL_NUMBER'])
df_after_drop = df_after_drop.drop(columns=['ORIGIN_AIRPORT'])
df_after_drop = df_after_drop.drop(columns=['DESTINATION_AIRPORT'])

df_after_drop = df_after_drop.drop(columns=['TAXI_OUT'])
df_after_drop = df_after_drop.drop(columns=['WHEELS_OFF'])

df_after_drop = df_after_drop.drop(columns=['ELAPSED_TIME'])
df_after_drop = df_after_drop.drop(columns=['AIR_TIME'])
df_after_drop = df_after_drop.drop(columns=['WHEELS_ON'])
df_after_drop = df_after_drop.drop(columns=['TAXI_IN'])



df_after_drop = df_after_drop.drop(columns=['DIVERTED'])
df_after_drop = df_after_drop.drop(columns=['CANCELLED'])
df_after_drop = df_after_drop.drop(columns=['CANCELLATION_REASON'])



# FILLING MISSING VALUES 

missing_values_series= df_after_drop.isnull().sum()

"""
YEAR	0
MONTH	0
DAY	0
DAY_OF_WEEK	0
DEPARTURE_DELAY	39515
DISTANCE	0
ARRIVAL_DELAY	43071
AIR_SYSTEM_DELAY	820047
SECURITY_DELAY	820047
AIRLINE_DELAY	820047
LATE_AIRCRAFT_DELAY	820047
WEATHER_DELAY	820047
delayreason	0

"""

def check_low_delay(row):
    
   if row["delayreason"]=="cancelled" :
       
       return 0
       
   else:
       return row["DEPARTURE_DELAY"]
   
def check_HIGH_delay(row):
    
   if row["delayreason"]=="cancelled" :
       
       return 0
   
   
   else:
       return row["ARRIVAL_DELAY"]
   
def cancelled_a(row):
    
   if (row["delayreason"]=="cancelled") :
    
       return 0
   else :
       return row["ARRIVAL_TIME"]
def cancelled_d(row):
    
   if(row["delayreason"]=="cancelled")  :
       
    
       return 0
   else :
       return row["DEPARTURE_TIME"]
   

          
          
     
          
      
          
    
    
    

      
    

    
    



df_after_drop['DEPARTURE_DELAY'] = df_after_drop.apply(check_low_delay,axis=1)


df_after_drop['ARRIVAL_DELAY'] = df_after_drop.apply(check_HIGH_delay,axis=1)


df_after_drop['ARRIVAL_TIME'] = df_after_drop.apply(cancelled_a,axis=1)


df_after_drop['DEPARTURE_TIME'] = df_after_drop.apply(cancelled_d,axis=1)


df_after_drop["ARRIVAL_DELAY"].fillna(df_after_drop["ARRIVAL_DELAY"].mean() ,inplace= True)





#df_after_drop['ARRIVAL_DELAY'] = df_after_drop.apply(final,axis=1)



print("ya rab yozbot")
#df_y = df_after_drop[df_after_drop["delayreason"]=="perfect"]
#df_z = df_after_drop[df_after_drop["delayreason"]=="AIRCRAFT"]
#df_a = df_after_drop[df_after_drop["delayreason"]=="AIRL"]
#df_b = df_after_drop[df_after_drop["delayreason"]=="AIRS"]
#df_c = df_after_drop[df_after_drop["delayreason"]=="SEC"]
#df_d = df_after_drop[df_after_drop["delayreason"]=="WEATHER"]

#x_mean=df_x.ARRIVAL_DELAY.mean()  # -12.446

#y_mean= df_y.ARRIVAL_DELAY.mean()  # 5.78

#z_mean= df_z.ARRIVAL_DELAY.mean()  #  64.4

#a_mean = df_a.ARRIVAL_DELAY.mean()  # 62.5
#b_mean = df_b.ARRIVAL_DELAY.mean()  # 62.5
#c_mean = df_c.ARRIVAL_DELAY.mean()  # 41.5
#d_mean = df_c.ARRIVAL_DELAY.mean()  # 41.5
mmm2= df_after_drop.isnull().sum()  



#### dropping unused colomns








df_after_drop = df_after_drop.drop(columns=['AIR_SYSTEM_DELAY'])
df_after_drop = df_after_drop.drop(columns=['SECURITY_DELAY'])
df_after_drop = df_after_drop.drop(columns=['AIRLINE_DELAY'])
df_after_drop = df_after_drop.drop(columns=['LATE_AIRCRAFT_DELAY'])
df_after_drop = df_after_drop.drop(columns=['WEATHER_DELAY'])

n=df_after_drop.isnull().sum()





##### machine learning 


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_after_drop['delayreason'] = le.fit_transform(df_after_drop['delayreason'])

x= df_after_drop.iloc[:,0:7].values 

y= df_after_drop.iloc[:,7:8].values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x= sc.fit_transform(x)


############

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=.2 , random_state=45)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

y_predict= lr.predict(x_test)

from sklearn.metrics import accuracy_score

lr_accuracy= accuracy_score(y_predict, y_test)


# 88% accuracy using logistic regression 



# support vector machine


from sklearn.svm import SVC 

sm = SVC()

sm.fit(x_train, y_train)
y_predict= sm.predict(x_test)
sm_accuracy= accuracy_score(y_predict, y_test)












