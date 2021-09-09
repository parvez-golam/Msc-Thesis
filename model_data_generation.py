
# Load libraries
from preprocessing import WIND_F
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Constants
DEMAND = 'Demand'
WIND = 'Wind'
WIND_G = 'Wind_G'
ZNORM = "zscore"
MINMAX_NORM = "minmax"

def normalize_data( 
    data, 
    method = ZNORM
):
    """
    Function to Normalize 'data' based on the 'method '
    """
    if method == ZNORM :
        data = data.reshape((len(data), 1))
        # train the standardization
        scaler = StandardScaler()
        scaler = scaler.fit(data)
        # standardization the dataset and print the first 5 rows
        normal_data = scaler.transform(data)
        normal_data = normal_data.flatten()
        

    elif method == MINMAX_NORM :
        data = data.reshape((len(data), 1))
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data)
        # normalize the dataset 
        normal_data = scaler.transform(data)
        normal_data = normal_data.flatten()

    elif method == "None":
        normal_data =  data
        scaler = None

    return normal_data , scaler

def denormalize_data(
    scaler, 
    data
):
    """
    Function to De-Normalize 'data'
    """
    if scaler:
        data = data.reshape((len(data), 1))
        denorm_data = scaler.inverse_transform(data)
        denorm_data = denorm_data.flatten()

    else:
        denorm_data = data
    
    return denorm_data


def df_to_array(
    df_ire_model, 
    typ
):
    """
    Function to convert Data frame to data array
    based on the 'typ'(wind/demand) of data 

    Returns Timesteps array and the data array
    """
    # data array - converting the dataframe to array 
    timesteps = df_ire_model.index.to_numpy()

    print("First-5 entry in data array :")
    print("------------------------------------------------------------")
    if typ == WIND :
        data_array = df_ire_model[WIND_G].to_numpy()
        print('Timestamp:\n', timesteps[:5], 
            '\nWind energy generation(MW):\n', data_array[:5])

    elif typ == DEMAND :
        data_array = df_ire_model[DEMAND].to_numpy()
        print('Timestamp:\n', timesteps[:5], 
            '\nEnergy Demand(MW):\n', data_array[:5])

    return timesteps, data_array

def _get_labelled_windows(
    x, 
    horizon=1
):
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(
    x, 
    window_size=20, 
    horizon=1
):
    """
    Function to view NumPy arrays as windows 
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    # print(f"Window step:\n {window_step}")

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    # create 2D array of windows of size window_size
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T 
    # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = _get_labelled_windows(x=windowed_array, horizon=horizon)
    print( '\nWindow shape:', windows.shape, '\nLabel shape:', labels.shape )

    return windows, labels

def print_window_label(
    windows, 
    labels, 
    n, 
    typ
):
    """
    Prints first 'n' matching pair of windows and labels
    """
    print("------------------------------------------------------------")
    print('\nFirst %s %s Window -> Label :' %(n, typ) )
    print("------------------------------------------------------------\n")
    for i in range(n):
        print(f"Window: {windows[i]} -> Label: {labels[i]}")

def make_train_test_splits(
    windows, 
    labels, 
    test_split=0.1
    ):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split)) # default to 90% train- 10% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    print(  'Train Window shape:', train_windows.shape, '\nTrain Label shape:', train_labels.shape,
        '\nTest Window shape:', test_windows.shape, '\nTest Label shape:', test_labels.shape )
            
    return train_windows, test_windows, train_labels, test_labels

def get_data_for_prediction(
    df_ire,
    scaler,
    year,
    prev_month,
    pred_month,
    window_size,
    horizon,
    typ
):
    """
    Function to generate data for every month's prediction
    """
    df_ire_prev_month = df_ire[(df_ire.index.year==year) & (df_ire.index.month== prev_month) ]
    if typ == WIND :
        prev_month = df_ire_prev_month[WIND_G].to_numpy()
    elif typ == DEMAND:
        prev_month = df_ire_prev_month[DEMAND].to_numpy()

    df_ire_month = df_ire[(df_ire.index.year==year) & (df_ire.index.month==pred_month) ]
    # data array
    timesteps = df_ire_month.index.to_numpy()
    if typ == WIND :
        actual_value = df_ire_month[WIND_G].to_numpy()
        eir_value = df_ire_month[WIND_F].to_numpy() # EIR Predictions 
    elif typ == DEMAND :
        actual_value = df_ire_month[DEMAND].to_numpy()



    prev_month_array  = prev_month.reshape((len(prev_month), 1))
    # Data normalization based on the 'normalization method'
    normalied_prev_month_array = scaler.transform(prev_month_array)
    normalied_prev_month_array = normalied_prev_month_array.flatten()

    # normalize 
    array_val = np.concatenate((prev_month[-window_size:], actual_value))
    array_val  = array_val.reshape((len(array_val ), 1))
    normalied_array = scaler.transform(array_val)
    normalied_array = normalied_array.flatten()

    windows, _ = make_windows(normalied_array, window_size=window_size, horizon=horizon)   

    if typ == WIND :
        ret_list = [windows, timesteps, actual_value, eir_value]
    elif typ == DEMAND :
        ret_list = [windows, timesteps,  actual_value, normalied_array]

    return  ret_list
