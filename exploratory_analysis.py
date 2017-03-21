import pandas as pd
import matplotlib  
import matplotlib.pyplot as plt  
import matplotlib.path as mplPath
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
import numpy as np
import seaborn as sns

df = pd.read_csv('2015_Green_Taxi_Trip_Data.csv') 

#print df.describe()

# List unique values in the df['pickup_datetime'] column
# df.pickup_datetime.unique()
 
# Remove some features (columns) from dataframe
df = df.drop('vendorid', 1)
df = df.drop('Extra', 1)
df = df.drop('MTA_tax', 1)
df = df.drop('Ehail_fee', 1)
df = df.drop('Improvement_surcharge', 1)
df = df.drop('Store_and_fwd_flag', 1)
df = df.drop('Payment_type', 1)

#-------------------------------------------------------------

import datetime as dt
df['Pickup_dt'] = df.pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%m/%d/%Y %I:%M:%S %p"))
# datetime format: 01/10/2015 05:24:59 PM

df['Pickup_hour'] = df.Pickup_dt.apply(lambda x:x.hour)
#print df.describe()
#print list(df)  # Print a list of all the dataframe columns :)

# Mean and Median of trip distance by pickup hour
# I will generate the table but also generate a plot for a better visualization
generate_trip_distance_graph = False
if generate_trip_distance_graph:
    fig,ax = plt.subplots(1,1,figsize=(9,5)) # prepare fig to plot mean and media values
    # use a pivot table to aggregate Trip_distance by hour
    table1 = df.pivot_table(index='Pickup_hour', values='Trip_distance',aggfunc=('mean','median',np.std)).reset_index()
    # rename columns
    table1.columns = ['Hour','Mean_distance','Median_distance','Std_distance']
    #table1[['Mean_distance','Median_distance']].plot(ax=ax)
    plt.errorbar(table1.Hour, table1.Mean_distance, table1.Std_distance, fmt='-o')
    plt.ylabel('Metric (miles)')
    plt.xlabel('Hours after midnight')
    plt.title('Distribution of trip distance by pickup hour')
    #plt.xticks(np.arange(0,30,6)+0.35,range(0,30,6))
    plt.xlim([0,23])
    plt.savefig('Question3_1.jpeg',format='jpeg')
    plt.show()
    print '-----Trip distance by hour of the day-----\n'
    print tabulate(table1.values.tolist(),["Hour","Mean distance","Median distance"])

generate_trip_distance_graph_day = False
if generate_trip_distance_graph_day:
    df['Pickup_day'] = df.Pickup_dt.apply(lambda x:x.day)
    fig,ax = plt.subplots(1,1,figsize=(9,5)) # prepare fig to plot mean and media values
    # use a pivot table to aggregate Trip_distance by hour
    table2 = df.pivot_table(index='Pickup_day', values='Trip_distance',aggfunc=('mean','median')).reset_index()
    # rename columns
    table2.columns = ['Day','Mean_distance','Median_distance']
    table2[['Mean_distance','Median_distance']].plot(ax=ax)
    plt.ylabel('Metric (miles)')
    plt.xlabel('Day')
    plt.title('Distribution of trip distance by pickup week day')
    #plt.xticks(np.arange(0,30,6)+0.35,range(0,30,6))
    plt.xlim([0,23])
    plt.savefig('distance_by_day.jpeg',format='jpeg')
    plt.show()

# Histogram of times
if False:
    df = df[df.Trip_distance < 20] 
    G = sns.distplot( df.Pickup_hour.dropna() ) #set_axis_labels( "Trip Distance","Counts")
    G.set_xlabel("Hours after midnight",size = 14,color="black")
    G.set_ylabel("Count",size = 14,color="black")
    G.tick_params(labelsize=12,labelcolor="black")
    plt.xlim([0,23])
    #G.figure.set_size_inches(14,9)
    sns.plt.show()

# Joint HexBin plot
if False:
    nstd=1.5
    df = df[df.Trip_distance <2.882150+nstd*2.947592]
    df = df[df.Trip_distance >2.882150-nstd*2.947592]
    df = df[df.Total_amount < 14.795530+nstd* 12.065141]
    df = df[df.Total_amount > 14.795530-nstd* 12.065141]
    jp1 = sns.jointplot( x="Trip_distance" , y="Total_amount" , data=df, kind="hex", color = 'k').set_axis_labels("Trip distance (miles)","Total amount (USD)")
    sns.plt.show()
    jp1.savefig('jointplot_dist_amount.png')
# Joint HexBin plot
if False:
    nstd=1
    df = df[df.Trip_distance <2.882150+nstd*2.947592]
    df = df[df.Trip_distance >2.882150-nstd*2.947592]
    df = df[df.Tip_amount < 1.1 + nstd*1.75]
    df = df[df.Tip_amount > 1.1 - nstd*1.75]
    df = df[df.Tip_amount > 0]
    jp1 = sns.jointplot(x="Trip_distance", y="Tip_amount",data=df,kind="hex",color='k').set_axis_labels("Trip distance (miles)","Tip amount (USD)")
    sns.plt.show()
    #jp1.savefig('jointplot_hours_amount.png')

# Correlation plot
if False:
    df['Pickup_day'] = df.Pickup_dt.apply(lambda x:x.day)
    #df = df.drop('pickup_datetime')
    #df = df.drop('dropoff_datetime')
    #df = df.drop('Pickup_dt')
    sns.linearmodels.corrplot(df, annot=False, diag_names=False)
    sns.plt.show()
    quit()

#-------------------AIRPORT TRIPS-------------------------------
# select airport trips
airports_trips = df[(df.rate_code==2) | (df.rate_code==3)]
print "Number of trips to/from NYC airports: ", airports_trips.shape[0]
print "Average fare (calculated by the meter) of trips to/from NYC airports: $", airports_trips.Fare_amount.mean(),"per trip"
print "Average total charged amount (before tip) of trips to/from NYC airports: $", airports_trips.Total_amount.mean(),"per trip"
# create a vector to contain Trip Distance for
v2 = airports_trips.Trip_distance # airport trips
v3 = df.loc[~df.index.isin(v2.index),'Trip_distance'] # non-airport trips
# remove outliers: exclude any data point located further than 3 standard deviations of the
# median point and  plot the histogram with 30 bins
v2 = v2[~((v2-v2.median()).abs()>3*v2.std())]
v3 = v3[~((v3-v3.median()).abs()>3*v3.std())]
# define bins boundaries
bins = np.histogram(v2,normed=True)[1]
h2 = np.histogram(v2,bins=bins,normed=True)
h3 = np.histogram(v3,bins=bins,normed=True)

# plot distributions of trip distance normalized among groups
#fig,ax = plt.subplots(1,2,figsize = (15,4))
fig,ax = plt.subplots(2,1,figsize = (8,10))
w = .4*(bins[1]-bins[0])
ax[0].bar(bins[:-1],h2[0],alpha=1,width=w,color='b')
ax[0].bar(bins[:-1]+w,h3[0],alpha=1,width=w,color='g')
ax[0].legend(['Airport trips','Non-airport trips'],loc='best',title='group')
ax[0].set_xlabel('Trip distance (miles)')
ax[0].set_ylabel('Group normalized trips count')
ax[0].set_title('A. Trip distance distribution')
#ax[0].set_yscale('log')

airports_trips.Pickup_hour.value_counts(normalize=True).sort_index().plot(ax=ax[1])
df.loc[~df.index.isin(v2.index),'Pickup_hour'].value_counts(normalize=True).sort_index().plot(ax=ax[1])
ax[1].set_xlabel('Hours after midnight')
ax[1].set_ylabel('Group normalized trips count')
ax[1].set_title('B. Hourly distribution of trips')
ax[1].legend(['Airport trips','Non-airport trips'],loc='best',title='group')
plt.savefig('Question3_2.jpeg',format='jpeg')
plt.show()
#---------------------------------------------------------------------



if False:
    nstd=2
    df = df[df.Trip_distance <2.882150+nstd*2.947592]
    df = df[df.Trip_distance >2.882150-nstd*2.947592]
    df = df[df.Tip_percentage >0]
    df = df[abs(df.Tip_percentage-df.Tip_percentage.mean() ) < 3*df.Tip_percentage.std() ]
    jp1 = sns.jointplot(x="Trip_distance",y="Tip_percentage",data=df,kind="hex",color='k').set_axis_labels("Trip_distance","Tip Percentage (%)")
    sns.plt.show()
    jp1.savefig('jointplot_dist_tip_percentage.png')
    quit()
if False:
    nstd=2
    df = df[df.Trip_distance <2.882150+nstd*2.947592]
    df = df[df.Trip_distance >2.882150-nstd*2.947592]
    df = df[df.Total_amount < 14.795530+nstd* 12.065141]
    df = df[df.Total_amount > 14.795530-nstd* 12.065141]
    jp1 = sns.jointplot( x="Trip_distance" , y="Total_amount" , data=df , kind="reg").set_axis_labels("Trip_distance","Total_amount")
    sns.plt.show()
    jp1.savefig('jointplot_dist_amount.png')


"""
CORRELATION TABLE

sns.linearmodels.corrplot(df, annot=False, diag_names=False)
sns.plt.show()
"""

"""

TRIP DISTANCE DISTRIBUTION PLOT
df = df[df.Trip_distance < 20] 
G = sns.distplot( df.Trip_distance.dropna() ) #set_axis_labels( "Trip Distance","Counts")
G.set_xlabel("Trip Distance (miles)",size = 14,color="black")
G.set_ylabel("Count",size = 14,color="black")
G.tick_params(labelsize=12,labelcolor="black")
#G.figure.set_size_inches(14,9)
sns.plt.show()
"""


"""
jp1 = sns.jointplot( x="Optical_GAP" , y="cGap" , data=df , kind="reg").set_axis_labels("Exp. optical gap", "b3lyp gap")
jp1.savefig('jointplot_Ogap_b3lypgap.png')
## Correlation Plot
sns.linearmodels.corrplot(df, annot=False, diag_names=False)
sns.plt.show()
"""



# df=pd.io.gbq.read_gbq("""  
# SELECT ROUND(pickup_latitude, 4) as lat, ROUND(pickup_longitude, 4) as long, COUNT(*) as num_pickups  
# FROM [nyc-tlc:yellow.trips_2015]  
# WHERE (pickup_latitude BETWEEN 40.61 AND 40.91) AND (pickup_longitude BETWEEN -74.06 AND -73.77 )  
# GROUP BY lat, long  
# """, project_id='taxi-1029')

df = df[df.Dropoff_latitude > 35]
df = df[df.Dropoff_longitude > -76]

pd.options.display.mpl_style = 'default' #Better Styling  
new_style = {'grid': False} #Remove grid  
matplotlib.rc('axes', **new_style)  
from matplotlib import rcParams  
rcParams['figure.figsize'] = (17.5, 17) #Size of figure  
rcParams['figure.dpi'] = 250

#P.set_axis_bgcolor('black') #Background Color

#P = df.plot(kind='scatter', x='Dropoff_longitude', y='Dropoff_latitude',color='red',xlim=(-74.06,-73.77),ylim=(40.61, 40.91),s=.02,alpha=.6)
P = df.plot(kind='scatter', x='Dropoff_longitude', y='Dropoff_latitude',color='red',xlim=(-74.06,-73.77),ylim=(40.61, 40.91),s=.02,alpha=.3)
P.set_axis_bgcolor('black')
plt.show()
