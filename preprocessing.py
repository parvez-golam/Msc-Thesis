
# Load libraries
import pandas as pd
import os
import numpy as np
import glob

# constants
DEMAND = 'Demand'
GENERATION = 'Generation'
WIND = 'Wind'
WIND_G = 'Wind_G'
WIND_F = 'Wind_F'
ALL = 'All'


def load_data( 
    path,
    year,
    typ='All'
):

    MONTH = ['01_Jan', '02_Feb', '03_Mar', '04_Apr', '05_May', '06_Jun', '07_Jul', '08_Aug', '09_Sep', '10_Oct', '11_Nov', '12_Dec']

    file_data_list = []
    # define file path
    for y in year:
        for m in MONTH:
            fpath = path + str(y) + "/" + m + "/"

            if typ == DEMAND:
                # get only demand files
                all_files = glob.glob( os.path.join(fpath, "*demand*.csv"))

            elif typ == WIND:
                # get only wind generation files
                all_files = glob.glob( os.path.join(fpath, "*wind*.csv"))

            elif typ == ALL :
                # get demand files
                demand_files = glob.glob( os.path.join(fpath, "*demand*.csv"))
                # get actual energy generation files
                gen_files = glob.glob( os.path.join(fpath, "*generation*.csv"))
                # get wind generation files
                wind_files = glob.glob( os.path.join(fpath, "*wind*.csv"))

                all_files = demand_files + gen_files + wind_files

            # get multiple files data in list of data frames
            for filename in all_files:
                df_file = pd.read_csv(filename, index_col=None, header=0)
                file_data_list.append(df_file)

    # get one single dataframe from dataframe list
    df = pd.concat(file_data_list)
    print("\n------------------------------------------------------------")
    print("\nShape of the data:")
    print(df.shape)
    print("\n------------------------------------------------------------")
    print(" First 5 rows: \n" )
    print(df.head())
    return df


def data_preprocess(df, typ):
    """
    Function to preprocess the data
    """

    if typ== ALL:
        # Merge all data 
        df = df.groupby([ ' REGION', 'DATE & TIME', ], as_index=False).first()


    # Drop a column if it does not have a value
    df = df.replace('-', np.nan)
    df = df.replace('NaN', np.nan)
    df = df.dropna(how='all')
    # print(df.shape)
    print("\n------------------------------------------------------------")
    print(" Column Names:" )
    print(df.columns)

    # convert date time 
    df['datetime'] = pd.to_datetime(df["DATE & TIME"])
    df['year'] = df['datetime'].dt.year

    print("\n------------------------------------------------------------")
    print("\n Dataset information:")
    print("--------------------------------------------------------------")
    print(df.info())

    # Get Ireland regions data
    if typ == WIND:
        # Remane columns of Wind data
        df = df.rename(columns={'  ACTUAL WIND(MW)': WIND_G,
                                ' FORECAST WIND(MW)': WIND_F})

        df.loc[:,WIND_G]=  pd.to_numeric(df[WIND_G])
        df.loc[:,WIND_F]=  pd.to_numeric(df[WIND_F])

        # Ireland data
        df_ire = df.loc[df[' REGION'] == 'Ireland'][[WIND_G, WIND_F, 'datetime']]

    elif typ == DEMAND:
        # Rename columns of Demand data
        df = df.rename(columns={' ACTUAL DEMAND(MW)': DEMAND})

        df.loc[:,DEMAND]=  pd.to_numeric(df[DEMAND])

        # Ireland data
        df_ire = df.loc[df[' REGION'] == 'Ireland'][[DEMAND, 'datetime']]

    elif typ == ALL:
        # Rename columns
        df = df.rename(columns={' ACTUAL DEMAND(MW)': DEMAND, 
                                ' ACTUAL GENERATION(MW)': GENERATION, 
                                ' FORECAST WIND(MW)' : WIND_F,
                                '  ACTUAL WIND(MW)' : WIND_G
                                }
                        )
        df.loc[:, DEMAND]=  pd.to_numeric(df[DEMAND])
        df.loc[:, WIND_F]=  pd.to_numeric(df[WIND_F])
        df.loc[:, WIND_G]=  pd.to_numeric(df[WIND_G])

        # Ireland data
        df_ire = df.loc[df[' REGION'] == 'Ireland']
    
    # Set datetime as index
    df_ire.set_index('datetime', inplace=True)

    print("-----------------------------------------------------")
    print("\n Ireland dataset information:")
    print("-----------------------------------------------------")
    print(df_ire.info())
    print("-----------------------------------------------------")

    # year wise count 
    print("\n-----------------------------------------------------")
    print("Year wise count")
    print("-----------------------------------------------------")
    print("\n Ireland")
    print(df_ire.index.year.value_counts())
    print("-----------------------------------------------------")

    if typ != ALL :
        # Drop missing values
        df_ire = df_ire.dropna() 

    return  df_ire, df

