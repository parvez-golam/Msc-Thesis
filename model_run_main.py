
# Load libraries
import json
import numpy as np
import preprocessing as pre
import models as mdl
import model_data_generation as dataGen
from itertools import product
from multiprocessing import Process, Manager
import tensorflow as tf
import time
# import os
# import pandas as pd
import csv
import gc

# Constants
ENERGY_DATA_PATH =  "\Path_of_energy_data"
MODEL_PARAMETERS_FILE = "parameters.json"
MODEL_SAVE_PATH = "model_experiments"
RESULTS_FILE = "results.csv"
MODELS =  ["rnn", "lstm"] # "gru"
DEMAND = 'Demand'
WIND = 'Wind'
# Used data till 2018 for model training 
YEARS = [ '2014', '2015', '2016', '2017', '2018'] #, '2019', '2020', '2021'
HORIZON = 25 # 6 hours data
WINDOW_SIZE = 30

def export_values(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for ins in product(*vals):
        yield dict(zip(keys, ins))


def load_parameters(param_file):
    """
    Function to load model execution hyper-parameters
    from .json file 'param_file'

    Returns execution hyper-parameters
    """
    # load the parameters for the models to run
    with open(param_file, "r") as params_file:
        params = json.load(params_file)

    return params

def get_gpu():
    """
    Function to allocate GPU
    """
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")

    for device in  gpu_devices :
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_visible_devices(device, "GPU")   


def run_model(
    mdl_ins,
    model_name,
    model_idx,
    dataset,
    norm_method,
    norm_scaler,
    test_split,
    learning_rate,
    model_args,
    train_windows, 
    train_labels, 
    test_windows, 
    test_labels, 
    batch_size,
    epochs,
    max_steps_per_epoch,
    return_dict
):
    """
    Function to rum model
    """

    tf.keras.backend.clear_session()
    # GPU
    get_gpu()

    # Model definition
    model = mdl_ins.create_model(model_name)(
                input_shape = WINDOW_SIZE,
                name = "%s_%s_%s%s_%s_%s" %(model_idx, dataset, model_name, batch_size, norm_method, test_split),
                output_size=HORIZON,
                optimizer= tf.optimizers.Adam(learning_rate=learning_rate),
                loss="mse",
                **model_args
            )
    print("\n------------------------------------------------------------")
    print("Model Summary : \n")
    print(model.summary())

    # Model training
    print('\nModel Training:')

    steps_per_epoch = min(
        int(np.ceil(train_windows.shape[0] / batch_size)), max_steps_per_epoch,
    )

    train_start_time = time.time()
    # Fit the model 
    fit = mdl_ins.fit_model(
            model, 
            train_windows, 
            train_labels, 
            test_windows, 
            test_labels, 
            batch_size,
            epochs, 
            steps_per_epoch,
            # split=test_split, 
            save_path= MODEL_SAVE_PATH
        )
    # training time
    training_time = time.time() - train_start_time

    # model validation
    print('\nModel Testing:')
    test_start_time = time.time()
    test_pred = mdl_ins.make_preds(model,test_windows).numpy()
    # prediction time
    test_time = time.time() - test_start_time

    # Model evaluation
    evaluation_results = mdl_ins.evaluate_preds(
                            y_true = dataGen.denormalize_data(norm_scaler, test_labels),
                            y_pred = dataGen.denormalize_data(norm_scaler, test_pred),
                        )
                            
    print(evaluation_results)

    results = [ model_name, dataset, model_idx,test_split, model_args['layers'], model_args['units'], model_args['layers'],
            HORIZON, batch_size,epochs, steps_per_epoch, "Adam", learning_rate, norm_method, training_time, test_time, 
            evaluation_results['mae'], evaluation_results['mse'], evaluation_results['rmse']
            ]
    # Store the results        
    return_dict['results'] = results 


    gc.collect()
    del model, test_pred
    

def main(
    data_path,
    param_file,
    result_file
):  
    """
    Main function to run the model experiments
    """
    print("------------------------------------------------------------")
    print("\nWind data:")
    # Load Wind data
    wind_data = pre.load_data(
                    path = data_path,
                    year = YEARS,
                    typ = WIND
                )
    wind_ire, wind_df = pre.data_preprocess(wind_data, typ= WIND)

    print("------------------------------------------------------------")
    print("\nDemand data:")
    # Load Demand data
    demand_data = pre.load_data(
                    path = data_path,
                    year = YEARS,
                    typ = DEMAND
                )
    demand_ire, demand_df = pre.data_preprocess(demand_data, typ=DEMAND)

    # dictionary with Ireland's Wind and Demand 
    dict_ire = { 
        WIND : wind_ire,
        DEMAND : demand_ire    
    }

    # load hyper-parameters
    params = load_parameters(param_file)

    # instance for model creation          
    mdl_ins = mdl.CreateModel()

    results =[]

    for k, v in dict_ire.items():
        print("\n------------------------------------------------------------")
        print("\n Models for %s Data: " %(k))
        print("\n------------------------------------------------------------")

        # convert data frame to data array
        timesteps , data_array = dataGen.df_to_array(df_ire_model = v, typ = k)

        for norm_method in params["normalization_method"] :

            print("\n------------------------------------------------------------")
            print("\n Data Normalization Method: %s" %(norm_method) )
            print("------------------------------------------------------------\n")
            # Data normalization based on the 'normalization method'
            normalied_data_array, norm_scaler = dataGen.normalize_data( 
                                                    data = data_array,
                                                    method = norm_method
                                                )

            # get windows and labels from the data array
            full_windows, full_labels = dataGen.make_windows(
                                            x = normalied_data_array, 
                                            window_size = WINDOW_SIZE, 
                                            horizon=HORIZON
                                        )
        
            # View the first 3 windows/labels
            dataGen.print_window_label(full_windows, full_labels, n=3, typ='full')

            # experiments_index = 0
            for test_split in params["test_split"]:
            
                print("\n------------------------------------------------------------")
                print("\n Test split: %s" %(test_split) )
                print("------------------------------------------------------------\n")
                # train test split of windows and labels
                train_windows, test_windows, train_labels, test_labels = dataGen.make_train_test_splits(
                                                                            windows = full_windows, 
                                                                            labels = full_labels, 
                                                                            test_split = test_split
                                                                        )
                # View the first 3 train-test windows/labels
                dataGen.print_window_label(train_windows, train_labels, n=3, typ='train')
                dataGen.print_window_label(test_windows, test_labels, n=3, typ='test')


                for epochs, batch_size, learning_rate in product(
                    params["epochs"],
                    params["batch_size"], 
                    params["learning_rate"]
                ):
                    for model_name in MODELS:
                        for model_idx, model_args in enumerate(
                            export_values(**params["model_params"][model_name])
                        ):

                            manager = Manager()
                            return_dict = manager.dict()

                            p = Process(
                                target = run_model,
                                args = (
                                    mdl_ins,
                                    model_name,
                                    model_idx,
                                    k,
                                    norm_method,
                                    norm_scaler,
                                    test_split,
                                    learning_rate,
                                    model_args,
                                    train_windows, 
                                    train_labels, 
                                    test_windows, 
                                    test_labels, 
                                    batch_size,
                                    epochs,
                                    params["max_steps_per_epoch"][0],
                                    return_dict
                                )
                            )
                            p.start()
                            p.join()

                            try:
                                results.append(return_dict['results']) 
                            except:
                                print("Model failed %s_%s_%s%s_%s_%s" %(model_idx, k, model_name, batch_size, norm_method, test_split))
                            
                            
    # csv fields                                
    fields = ["MODEL","DATA_TYP", "MODEL_INDEX","TEST_SPLIT", "LAYERS","UNITS", "RETURN_SEWQUENCE", "FORECAST_HORIZON","BATCH_SIZE",
        "EPOCHS","STEPS", "OPTIMIZER","LEARNING_RATE","NORMALIZATION","TRAINING_TIME","TEST_TIME",'MAE','MSE','RMSE']

    # writing to csv file 
    f = open(result_file, 'w', newline='') 
    # create csv writer 
    writer =  csv.writer(f) 
    # writing headers (field names) 
    writer.writerow(fields) 
    writer.writerows(results)
    # close the file
    f.close() 


if __name__ == "__main__":

    main(
        data_path = ENERGY_DATA_PATH,
        param_file = MODEL_PARAMETERS_FILE,
        result_file = RESULTS_FILE
    )                                                                                                                     
