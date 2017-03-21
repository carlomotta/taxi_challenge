# Function to run the feature engineering
import datetime as dt
import numpy as np
import matplotlib.path as mplPath
def engineer_features(adf):
    """
    This function create new variables based on present variables in the dataset adf. It creates:
    . Week: int {1,2,3,4,5}, Week a transaction was done
    . Week_day: int [0-6], day of the week a transaction was done
    . Month_day: int [0-30], day of the month a transaction was done
    . Hour: int [0-23], hour the day a transaction was done
    . Shift type: int {1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)}, shift of the day  
    . Speed_mph: float, speed of the trip
    . Tip_percentage: float, target variable
    . With_tip: int {0,1}, 1 = transaction with tip, 0 transction without tip
    
    input:
        adf: pandas.dataframe
    output: 
        pandas.dataframe
    """
    
    # make copy of the original dataset
    df = adf.copy()
    
    # derive time variables
    print "deriving time variables..."
    ref_week = dt.datetime(2015,9,1).isocalendar()[1] # first week of september in 2015
    df['Week'] = df.Pickup_dt.apply(lambda x:x.isocalendar()[1])-ref_week+1
    df['Week_day']  = df.Pickup_dt.apply(lambda x:x.isocalendar()[2])
    df['Month_day'] = df.Pickup_dt.apply(lambda x:x.day)
    df['Hour'] = df.Pickup_dt.apply(lambda x:x.hour)
    #df.rename(columns={'Pickup_hour':'Hour'},inplace=True)

    # create shift variable:  1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)
    df['Shift_type'] = np.NAN
    df.loc[df[(df.Hour>=7) & (df.Hour<15)].index,'Shift_type'] = 1
    df.loc[df[(df.Hour>=15) & (df.Hour<23)].index,'Shift_type'] = 2
    df.loc[df[df.Shift_type.isnull()].index,'Shift_type'] = 3
    
    # Trip duration 
    print "deriving Trip_duration..."
    df['Trip_duration'] = ((df.Dropoff_dt-df.Pickup_dt).apply(lambda x:x.total_seconds()/60.))
    
    print "deriving direction variables..."
    # create direction variable Direction_NS. 
    # This is 2 if taxi moving from north to south, 1 in the opposite direction and 0 otherwise
    df['Direction_NS'] = (df.Pickup_latitude>df.Dropoff_latitude)*1+1
    indices = df[(df.Pickup_latitude == df.Dropoff_latitude) & (df.Pickup_latitude!=0)].index
    df.loc[indices,'Direction_NS'] = 0

    # create direction variable Direction_EW. 
    # This is 2 if taxi moving from east to west, 1 in the opposite direction and 0 otherwise
    df['Direction_EW'] = (df.Pickup_longitude>df.Dropoff_longitude)*1+1
    indices = df[(df.Pickup_longitude == df.Dropoff_longitude) & (df.Pickup_longitude!=0)].index
    df.loc[indices,'Direction_EW'] = 0
    
    # create variable for Speed
    print "deriving Speed. Make sure to check for possible NaNs and Inf vals..."
    df['Speed_mph'] = df.Trip_distance/(df.Trip_duration/60)
    # replace all NaNs values and values >240mph by a values sampled from a random distribution of 
    # mean 12.9 and  standard deviation 6.8mph. These values were extracted from the distribution
    indices_oi = df[(df.Speed_mph.isnull()) | (df.Speed_mph>240)].index
    df.loc[indices_oi,'Speed_mph'] = np.abs(np.random.normal(loc=12.9,scale=6.8,size=len(indices_oi)))
    print "Feature engineering done! :-)"
    
    # Create a new variable to check if a trip originated in Upper Manhattan
    print "checking where the trip originated..."
    df['U_manhattan'] = df[['Pickup_latitude','Pickup_longitude']].apply(lambda r:is_within_polygon((r[-1],r[1])),axis=1)
    
    # create tip percentage variable
    df['Tip_percentage'] = 100*df.Tip_amount/df.Total_amount
    
    # create with_tip variable
    df['With_tip'] = (df.Tip_percentage>0)*1

    return df

umanhattan = \
[[40.796937, -73.949503],[40.787945, -73.955822],[40.782772, -73.943575],\
 [40.794715, -73.929801],[40.811261, -73.934153],[40.835371, -73.934515],\
 [40.868910, -73.911145],[40.872719, -73.910765],[40.878252, -73.926350],\
 [40.850557, -73.947262],[40.836225, -73.949899],[40.806050, -73.971255]]
def is_within_polygon(point,polygon=umanhattan):
    """
    Inputs: point = [x,y]
            polygon = [[x1,y1],[x2,y2],...,[xn,yn]]
    """
    bbPath = mplPath.Path(np.array( polygon ))
    return bbPath.contains_point((point[0],point[1]))

