from tensorflow.keras import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import keras_tuner

class Classifyer:
    def __init__(this):
        # Define CNN architecture
         this.model = Sequential()
    
    def build_model(this,hp):
       
        num_layers = 9
        num_epochs = 10
        num_batch = 64
        # Convolutional layers
        this.model.add(Conv2D(filters=hp.Int('units_0', min_value=32, max_value = 512, step=128), kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        this.model.add(MaxPooling2D((2, 2)))
        this.model.add(Conv2D(filters=hp.Int('units_1', min_value=32, max_value = 512, step=128), kernel_size=(3, 3), activation='relu'))
        this.model.add(MaxPooling2D((2, 2)))
        this.model.add(Conv2D(filters=hp.Int('units_2', min_value=32, max_value = 64, step=16), kernel_size=(3, 3), activation='relu'))
        # Flatten layer
        this.model.add(Flatten())
        
        # Fully connected layers
        this.model.add(Dense(units=hp.Int('dense', min_value=32, max_value = 64, step=16), activation='relu'))
        this.model.add(Dense(3, activation='softmax'))  # Output layer with 3 classes for the superclasses in fashion MNIST
        
        # Compile the model
        this.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',metrics.Precision(),metrics.Recall(),metrics.CategoricalAccuracy()])
        
        return this.model

    def setup_model(this,train_x, train_y, val_x, val_y):

        print(f"Training data shape: {train_x.shape}, {train_y.shape}")
        print(f"Validation data shape: {val_x.shape}, {val_y.shape}")
        #hp = keras_tuner.HyperParameters()
        #m = this.build_model(hp)
        hp_tuner = keras_tuner.RandomSearch(this.build_model, objective='val_accuracy', max_trials=5,executions_per_trial=2, overwrite=True)
        hp_tuner.search_space_summary()
        
        this.hp_tuner = hp_tuner

    def parameter_search(this, train_x, train_y, val_x, val_y):
        print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)
        this.hp_tuner.search(train_x, train_y, epochs=10, validation_data=(val_x, val_y))

    def train_model(this, train_x, train_y, val_x, val_y):
        best_hps = this.hp_tuner.get_best_hyperparameters(5)
        this.model = this.build_model(best_hps[0])
        this.history = this.model.fit(train_x,train_y, epochs=10, validation_data=(val_x,val_y))

    def predict_data(this, test_x, test_y):
        this.y_predicted = []
        this.error_list = []
        this.confidences = []
        for i, (X, y) in enumerate(zip(test_x, test_y)):
            y_pred = this.model.predict(X.reshape(1, 28, 28, 1), verbose=0)
            confidence = np.max(y_pred)
            this.confidences.append(confidence)
            y_pred_class = np.argmax(y_pred)
            this.y_predicted.append(y_pred_class)
            y_class = np.argmax(y)
            err = 1 - (y_pred_class.item() == y_class.item())
            this.error_list.append(err)
