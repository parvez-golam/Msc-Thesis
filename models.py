#-----------------Model creation----------------

# Load libraries
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
DEMAND = 'Demand'
WIND = 'Wind'
WIND_G = 'Wind_G'

"""
   Class definition to implement Deep learning models for Wind project.
   
"""
class CreateModel:
    """
    Class to create models and predict from Model.
    """

    def __init__(self):
        """
        Create 
        """ 

    def _create_model_checkpoint(
        self, 
        model_name, 
        save_path="model_experiments"
    ):
        """
        Method to implement a ModelCheckpoint callback with a filename
        """
        return tf.keras.callbacks.ModelCheckpoint(  filepath= os.path.join(save_path, model_name), 
                                                    verbose=0, 
                                                    save_best_only=True  )
 

    def fit_model(
        self, 
        model, 
        train_windows, 
        train_labels, 
        test_windows, 
        test_labels,
        batch_size,
        epochs,
        max_steps_per_epoch,
        save_path
    ):
        """
        Method to fit the model 
        """
        steps_per_epoch = min(
            int(np.ceil(train_windows.shape[0] / batch_size)), max_steps_per_epoch
        )

        fit = model.fit(train_windows,
                        train_labels,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=0,
                        batch_size=batch_size,
                        validation_data=(test_windows, test_labels),
                        callbacks=[self._create_model_checkpoint(model_name=model.name, save_path= save_path)])

        return fit  

    def _compile_model(
        self, 
        model, 
        loss, 
        optimizer
    ):
        """
        Method to compile model
        """
        model.compile( loss=loss, optimizer=optimizer) 


    def rnn(
        self,
        input_shape,
        name,
        output_size=1,
        optimizer=tf.keras.optimizers.Adam() ,
        loss="mse",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
        typ=None,
    ):
        """
        Definition of RNN model
        """
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)

        # RNN layers
        return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
        x = tf.keras.layers.SimpleRNN(
            recurrent_units[0],
            return_sequences=return_sequences_tmp,
            dropout=recurrent_dropout,
        )(x)
        for i, u in enumerate(recurrent_units[1:]):
            return_sequences_tmp = (
                return_sequences if i == len(recurrent_units) - 2 else True
            )
            x = tf.keras.layers.SimpleRNN(
                u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
            )(x)

        # Dense layers
        if return_sequences:
            x = tf.keras.layers.Flatten()(x)
        for hidden_units in dense_layers:
            x = tf.keras.layers.Dense(hidden_units)(x)
            if dense_dropout > 0:
                x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
        x = tf.keras.layers.Dense(output_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)

        # Compile model
        self._compile_model(model=model, loss=loss, optimizer=optimizer)

        return model

    def lstm(
        self,
        input_shape,
        name,
        output_size=1,
        optimizer=tf.keras.optimizers.Adam() ,
        loss="mse",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
        typ=None,
    ):
        """
        Definition of LSTM models
        """
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)

        # LSTM layers
        return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
        x = tf.keras.layers.LSTM(
            recurrent_units[0],
            return_sequences=return_sequences_tmp,
            dropout=recurrent_dropout,
        )(x)
        for i, u in enumerate(recurrent_units[1:]):
            return_sequences_tmp = (
                return_sequences if i == len(recurrent_units) - 2 else True
            )
            x = tf.keras.layers.LSTM(
                u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
            )(x)

        # Dense layers
        if return_sequences:
            x = tf.keras.layers.Flatten()(x)
        for hidden_units in dense_layers:
            x = tf.keras.layers.Dense(hidden_units)(x)
            if dense_dropout > 0:
                x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
        x = tf.keras.layers.Dense(output_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)

        # Compile model
        self._compile_model(model=model, loss=loss, optimizer=optimizer)

        return model

    def create_model(
        self,
        func_name
    ):  
        """
        """
        model_dict = {   
            "rnn": self.rnn,
            "lstm": self.lstm
        }

        func = model_dict[func_name]

        return lambda input_shape, name, output_size, optimizer, loss, **args: func(
            input_shape=input_shape,
            name = name,
            output_size=output_size,
            optimizer=optimizer,
            loss="mse",
            recurrent_units=[args["units"]] * args["layers"],
            return_sequences=args["return_sequence"]
        )
        

    def make_preds(
        self, 
        model, 
        input_data
    ):
        """
        Method that uses 'model' to make predictions on 'input_data'.

        Parameters
        ----------
        model: trained model 
        input_data: windowed input data (same kind of data model was trained on)

        Returns model predictions on input_data.
        """
        forecast = model.predict(input_data)
        return tf.squeeze(forecast) # return 1D array of predictions                                         

    def evaluate_preds(
        self, 
        y_true, 
        y_pred
    ):
        """
        Method to evaluate prediction values with True values.

        Parameters
        ----------
        y_true: True values 
        y_pred: Predicted values

        Returns difference between prediction values and True values.
        """  
        # various metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred) 
        rmse = np.sqrt(mse)

        return {"mae": mae,
                "mse": mse,
                "rmse": rmse
                }
  