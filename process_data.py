import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class Data: 
    def __init__(this):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (val_images, val_labels) = fashion_mnist.load_data()
        
        this.train_images_original = train_images
        this.train_labels_original = train_labels
        this.val_images_original = val_images
        this.val_labels_original = val_labels
        this.original_labels_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def remove_class_from_training_data(this, remove_class):
         #boolean mask to a remove class from training data
        this.removed_class = remove_class
        mask = this.train_labels_original != remove_class
        
        this.train_x = this.train_images_original[mask]
        this.train_y = this.train_labels_original[mask]
        
        #keep class array
        mask = this.train_labels_original == remove_class
        removed_images = this.train_images_original[mask]
        removed_labels = this.train_labels_original[mask]
        
        ## repeat for validation set (used to verify base learner) 
        
        # boolean mask to remove class 2 from validation data
        mask = this.val_labels_original != remove_class
        this.val_x = this.val_images_original[mask]
        this.val_y = this.val_labels_original[mask]
        
        #keep class array
        mask = this.val_labels_original == remove_class
        
        val_removed_images = this.val_images_original[mask]
        val_removed_labels = this.val_labels_original[mask]
        
        # store all removed images and labels in the same arrays
        this.removed_class_images = np.concatenate([removed_images, val_removed_images])
        this.removed_class_labels = np.concatenate([removed_labels, val_removed_labels])
        
    def simulation_split(this, test_size):
        # Split the original training data into training and test datasets for simulation
        train_images, test_images, train_labels, test_labels = train_test_split(this.train_x, this.train_y, test_size=test_size, random_state=42)
        this.train_x_split = train_images
        this.train_y_split = train_labels
        this.test_x_split = test_images
        this.test_y_split = test_labels
        
    def preprocess_class_names(this):
        clothes = [0,1,2,3,4,6]
        shoes = [5,7,9]
        acc = [8]
        this.new_class_names = ["Clothes", "Shoes", "Accessories"]
        
        this.train_y_super = np.copy(this.train_y_split)
        # change labels in training set to new super class
        clothes_mask = np.isin(this.train_y_split, clothes)
        this.train_y_super[clothes_mask] = 0
        shoes_mask = np.isin(this.train_y_split, shoes)
        this.train_y_super[shoes_mask] = 1
        acc_mask = np.isin(this.train_y_split, acc)
        this.train_y_super[acc_mask] = 2
        
        #  change labels in validation set 
        this.val_y_super = np.copy(this.val_y)
        clothes_mask = np.isin(this.val_y, clothes)
        this.val_y_super[clothes_mask] = 0
        shoes_mask = np.isin(this.val_y, shoes)
        this.val_y_super[shoes_mask] = 1
        acc_mask = np.isin(this.val_y, acc)
        this.val_y_super[acc_mask] = 2

        #  change labels in the test set
        this.test_y_super = np.copy(this.test_y_split)
        clothes_mask = np.isin(this.test_y_split, clothes)
        this.test_y_super[clothes_mask] = 0
        shoes_mask = np.isin(this.test_y_split, shoes)
        this.test_y_super[shoes_mask] = 1
        acc_mask = np.isin(this.test_y_split, acc)
        this.test_y_super[acc_mask] = 2

        this.removed_class_labels_super = np.copy(this.removed_class_labels)
        # also change the removed class to the new superclass 
        if this.removed_class in clothes: 
            this.removed_class_labels_super[:] = 0
        elif this.removed_class in shoes: 
            this.removed_class_labels_super[:] = 1
        elif this.removed_class in acc: 
            this.removed_class_labels_super[:] = 2

    def preprocess(this):
        clothes = [0,1,2,3,4,6]
        shoes = [5,7,9]
        acc = [8]
        # reshape data to single channel
        this.train_x_split = this.train_x_split.reshape((this.train_x_split.shape[0], 28, 28, 1))
        this.val_x = this.val_x.reshape((this.val_x.shape[0], 28, 28, 1))
        this.test_x_split = this.test_x_split.reshape((this.test_x_split.shape[0], 28, 28, 1))
        this.removed_class_images = this.removed_class_images.reshape((this.removed_class_images.shape[0], 28, 28, 1))
        
        # one hot encode target vals
        this.train_y_super = to_categorical(this.train_y_super)
        this.val_y_super = to_categorical(this.val_y_super)
        this.test_y_super = to_categorical(this.test_y_super)
        
        # "one hot encode" removed class - need to add the rest of the of the labels to the array
        # to have the same format as the training labels for example [0 0 1]
        one_hot_removed = np.zeros((len(this.removed_class_labels_super),3))
        one_hot_removed[np.arange(len(this.removed_class_labels_super)), this.removed_class_labels_super] = 1
        this.removed_class_labels_super = one_hot_removed



        # normalize pixel data
        # convert from integers to floats
        this.train_x_split = this.train_x_split.astype('float32')
        this.val_x = this.val_x.astype('float32')
        this.removed_class_images = this.removed_class_images.astype('float32')
        this.test_x_split = this.test_x_split.astype('float32')
        # normalize to range 0-1
        this.train_x_split = this.train_x_split / 255.0
        this.val_x = this.val_x / 255.0
        this.test_x_split = this.test_x_split / 255.0
        this.removed_class_images = this.removed_class_images/ 255.0


    def concat_test_data(this):
        # split images and labels into subsets, doing this we can control when and where to introduce drift
        test_x_1 = this.test_x_split[:2000]
        test_y_1 = this.test_y_super[:2000]
        test_original_y_1 = this.test_y_split[:2000]
        
        test_x_2 = this.test_x_split[2000:10000]
        test_y_2 = this.test_y_super[2000:10000]
        test_original_y_2 = this.test_y_split[2000:10000]
        
        test_x_3 = this.test_x_split[10000:18000]
        test_y_3 = this.test_y_super[10000:18000]
        test_original_y_3 = this.test_y_split[10000:18000]
        
        test_x_4 = this.test_x_split[18000:]
        test_y_4 = this.test_y_super[18000:]
        test_original_y_4 = this.test_y_split[18000:]        


        # split removed class images and labels into subsets, so we can introduce multiple drift points
        removed_x_1 = this.removed_class_images[:3500]
        removed_y_1 = this.removed_class_labels_super[:3500]
        removed_original_y_1 = this.removed_class_labels[:3500]
        
        removed_x_2 = this.removed_class_images[3500:]
        removed_y_2 = this.removed_class_labels_super[3500:]
        removed_original_y_2 = this.removed_class_labels[3500:]

        
        # reintroduce removed class in two intervalls, with subsets 2 and 4
        reintroduce_x_1 = np.concatenate((removed_x_1,test_x_2,), axis=0)
        reintroduce_y_1 = np.concatenate((removed_y_1,test_y_2,), axis=0)
        reintroduce_y_original_1 = np.concatenate((removed_original_y_1,test_original_y_2,), axis=0)

        reintroduce_x_2 = np.concatenate((removed_x_2, test_x_4), axis=0)
        reintroduce_y_2 = np.concatenate((removed_y_2, test_y_4), axis=0)
        reintroduce_y_original_2 = np.concatenate((removed_original_y_2, test_original_y_4), axis=0)
        
        
        shuffle_indices_1 =  np.random.permutation(len(reintroduce_x_1))
        
        # Shuffle the data arrays using the permutation indices
        shuffled_x_1 = reintroduce_x_1[shuffle_indices_1]
        shuffled_y_1 = reintroduce_y_1[shuffle_indices_1]
        shuffled_y_original_1 = reintroduce_y_original_1[shuffle_indices_1]

                
        shuffle_indices_2 =  np.random.permutation(len(reintroduce_x_2))
        
        # Shuffle the data arrays using the permutation indices
        shuffled_x_2 = reintroduce_x_2[shuffle_indices_2]
        shuffled_y_2 = reintroduce_y_2[shuffle_indices_2]
        shuffled_y_original_2 = reintroduce_y_original_2[shuffle_indices_2]
        
        # put all subsets together
        stream_x = np.concatenate((test_x_1,shuffled_x_1), axis=0)
        stream_x = np.concatenate((stream_x,test_x_3), axis=0)
        stream_x = np.concatenate((stream_x, shuffled_x_2), axis=0)
        
        stream_y = np.concatenate((test_y_1,shuffled_y_1), axis=0)
        stream_y = np.concatenate((stream_y,test_y_3), axis=0)
        stream_y = np.concatenate((stream_y, shuffled_y_2), axis=0)

                
        stream_y_original = np.concatenate((test_original_y_1,shuffled_y_original_1), axis=0)
        stream_y_original = np.concatenate((stream_y_original,test_original_y_3), axis=0)
        stream_y_original = np.concatenate((stream_y_original, shuffled_y_original_2), axis=0)

        this.stream_x = stream_x
        this.stream_y = stream_y
        this.stream_y_original = stream_y_original

         # drift time 1: first occurence of the removed label 

        drift1 = len(test_x_1) + np.where(shuffled_y_original_1 == this.removed_class)[0][0]
        print(drift1,  np.where(shuffled_y_original_1 == this.removed_class)[0][0])
        # drift time 2: after the last occurence of removed label in the first drift interval (returning to original distribution would be considered a gradual or recurring drift)
        drift2 = len(test_x_1)+ max(index for index, item in enumerate(shuffled_y_original_1) if item == this.removed_class)
        print(drift2)
        # drift time 3: first occurence in the second drift interval 
        drift3 = len(test_x_1) + len(shuffled_x_1) + len(test_x_3) + np.where(shuffled_y_original_2 == this.removed_class)[0][0]
        print(drift3, np.where(shuffled_y_original_2 == this.removed_class)[0][0])
        # drift time 4: last occurence in the second drift interval
        drift4 = len(test_x_1) + len(shuffled_x_1) + len(test_x_3)+  max(index for index, item in enumerate(shuffled_y_original_2) if item == this.removed_class)
        print(drift4)
        this.drift1 = drift1
        this.drift2 = drift2
        this.drift3 = drift3
        this.drift4 = drift4
        

