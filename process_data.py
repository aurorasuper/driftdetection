import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class Data: 
    def __init__(this):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (val_images, val_labels) = fashion_mnist.load_data()
        
        # this.train_images_original = train_images
        # this.train_labels_original = train_labels
        # this.val_images_original = val_images
        # this.val_labels_original = val_labels
        this.all_data_images = np.concatenate((train_images, val_images))
        this.all_data_labels = np.concatenate((train_labels, val_labels))
        this.original_labels_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def remove_class_from_training_data(this, remove_class):

        this.removed_class = remove_class
        #keep class array
        mask_keep = this.all_data_labels == remove_class
        this.removed_images = this.all_data_images[mask_keep]
        this.removed_labels = this.all_data_labels[mask_keep]
      
        mask_remove = this.all_data_labels != remove_class
        this.all_data_images = this.all_data_images[mask_remove]
        this.all_data_labels = this.all_data_labels[mask_remove]

        
        
    def generate_drift_simulation(this, simulation_size):
        # Split the original training data into training and test datasets for simulation
        train_images, simulation_images, train_labels, simulation_labels = train_test_split(this.all_data_images, this.all_data_labels, 
                                                                                            test_size=simulation_size, random_state=42)
        p_test_val = 0.1
        train_set_length = len(train_images)
        n_test_val = int(train_set_length*p_test_val)
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=n_test_val, random_state=42)
        
        train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=n_test_val, random_state=42)

        this.train_images = train_images
        this.train_labels = train_labels
        this.test_images = test_images
        this.test_labels = test_labels
        this.val_images = val_images
        this.val_labels = val_labels
        this.simulation_images = simulation_images
        this.simulation_labels = simulation_labels
        
    def preprocess_class_names(this):
        clothes = [0,1,2,3,4,6]
        shoes = [5,7,9]
        acc = [8]
        this.new_class_names = ["Clothes", "Shoes", "Accessories"]
        train_labels_super = np.copy(this.train_labels)
        test_labels_super = np.copy(this.test_labels)
        val_labels_super = np.copy(this.val_labels)
        simulation_labels_super = np.copy(this.simulation_labels)
        super_labels_list = [(this.train_labels, train_labels_super),
                             (this.test_labels, test_labels_super), 
                             (this.val_labels, val_labels_super),
                             (this.simulation_labels, simulation_labels_super)
                            ]
        for original_labels, super_labels in super_labels_list: 
            # change labels in training set to new super class
            clothes_mask = np.isin(original_labels, clothes)
            super_labels[clothes_mask] = 0
            shoes_mask = np.isin(original_labels, shoes)
            super_labels[shoes_mask] = 1
            acc_mask = np.isin(original_labels, acc)
            super_labels[acc_mask] = 2

        this.train_labels_super = super_labels_list[0][1]
        this.test_labels_super = super_labels_list[1][1]
        this.val_labels_super = super_labels_list[2][1]
        this.simulation_labels_super = super_labels_list[3][1]
        
        this.removed_labels_super = np.copy(this.removed_labels)
        # also change the removed class to the new superclass 
        if this.removed_class in clothes: 
            this.removed_labels_super[:] = 0
        elif this.removed_class in shoes: 
            this.removed_labels_super[:] = 1
        elif this.removed_class in acc: 
            this.removed_labels_super[:] = 2

        print(this.removed_labels_super.shape)

    def preprocess(this):
        clothes = [0,1,2,3,4,6]
        shoes = [5,7,9]
        acc = [8]
        # reshape data to single channel
        this.train_images = this.train_images.reshape((this.train_images.shape[0], 28, 28, 1))
        this.test_images = this.test_images.reshape((this.test_images.shape[0], 28, 28, 1))
        this.val_images = this.val_images.reshape((this.val_images.shape[0], 28, 28, 1))
        this.simulation_images = this.simulation_images.reshape((this.simulation_images.shape[0], 28, 28, 1))
        this.removed_images = this.removed_images.reshape((this.removed_images.shape[0], 28, 28, 1))
        
        # one hot encode target vals
        this.train_labels_super = to_categorical(this.train_labels_super)
        this.test_labels_super = to_categorical(this.test_labels_super)
        this.val_labels_super = to_categorical(this.val_labels_super)
        this.simulation_labels_super = to_categorical(this.simulation_labels_super)
        
        # "one hot encode" removed class - need to add the rest of the of the labels to the array
        # to have the same format as the training labels for example [0 0 1]
        one_hot_removed = np.zeros((len(this.removed_labels_super),3))
        one_hot_removed[np.arange(len(this.removed_labels_super)), this.removed_labels_super] = 1
        this.removed_labels_super = one_hot_removed



        # normalize pixel data
        # convert from integers to floats
        this.train_images = this.train_images.astype('float32')
        this.test_images = this.test_images.astype('float32')
        this.val_images = this.val_images.astype('float32')
        this.removed_images = this.removed_images.astype('float32')
        this.simulation_images = this.simulation_images.astype('float32')
        # normalize to range 0-1
        this.train_images = this.train_images / 255.0
        this.test_images = this.test_images / 255.0
        this.val_images = this.val_images / 255.0
        this.simulation_images = this.simulation_images / 255.0
        this.removed_images = this.removed_images/ 255.0

    def shuffle_data(this, images, labels, superlabels):
        shuffle_indices =  np.random.permutation(len(images))
        shuffled_images = images[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]
        shuffled_labels_super = superlabels[shuffle_indices]
        return shuffled_images, shuffled_labels,shuffled_labels_super

    def split_data(this,start, end, images, labels, superlabels):
        
        split_images = images[start:end]
        split_labels = labels[start:end]
        split_superlabels = superlabels[start:end]
        return split_images, split_labels, split_superlabels

    def merge_data(this, images_1, images_2, labels_1, labels_2, superlabels_1, superlabels_2):
        merged_images = np.concatenate((images_1, images_2))
        merged_labels = np.concatenate((labels_1, labels_2))
        merged_superlabels = np.concatenate((superlabels_1,superlabels_2))
        return merged_images, merged_labels, merged_superlabels
        
    def concat_test_data(this):
        clothes = [0,1,2,3,4,6]
        shoes = [5,7,9]
        acc = [8]
        # split images and labels into subsets, doing this we can control when and where to introduce drift
        simulation_length = len(this.simulation_labels)
        simulation_length_quarter = int(simulation_length // 4)

        exlusion_set_images, exlusion_set_labels, exlusion_set_labels_super = this.split_data(0, simulation_length_quarter, 
                                                                                      this.simulation_images,
                                                                                      this.simulation_labels, this.simulation_labels_super)

        include_set_images, include_set_labels, include_set_labels_super = this.split_data(simulation_length_quarter, simulation_length, 
                                                                                      this.simulation_images,
                                                                                      this.simulation_labels, this.simulation_labels_super)

        inclusion_set_length = len(include_set_labels_super) 
        inclusion_set_third = int(inclusion_set_length //3)
        include_images_1, include_labels_1, include_labels_super_1 = this.split_data(0, simulation_length_quarter, 
                                                                                      include_set_images,
                                                                                      include_set_labels, include_set_labels_super)

        include_images_2, include_labels_2, include_labels_super_2 = this.split_data(simulation_length_quarter, simulation_length_quarter*2, 
                                                                                      include_set_images,
                                                                                      include_set_labels, include_set_labels_super)
        
        include_images_3, include_labels_3, include_labels_super_3 = this.split_data(simulation_length_quarter*2, simulation_length_quarter*3, 
                                                                              include_set_images,
                                                                              include_set_labels, include_set_labels_super)

        removed_super_class = 0
        num_category_of_removed = 0
        if this.removed_class in clothes: 
            removed_super_class = 0
        elif this.removed_class in shoes: 
            removed_super_class = 1
        elif this.removed_class in acc: 
            removed_super_class = 2

        print("Removed super class is: ", removed_super_class)
        # count the number of instances of the super class to which the removed class belongs
        if removed_super_class == 0: 
            num_category_of_removed = np.count_nonzero(include_set_labels == 0)
        elif removed_super_class == 1: 
            num_category_of_removed = np.count_nonzero(include_set_labels == 1)
        elif removed_super_class == 2: 
            num_category_of_removed = np.count_nonzero(include_set_labels == 2)

        
        print(num_category_of_removed)
        # sets increasing drift severiy by increasing the number of removed class in different intervals
        n_include_removed_1 = int(num_category_of_removed * 0.2 )
        n_include_removed_2 = int(num_category_of_removed * 0.4 )
        n_include_removed_3 = int(num_category_of_removed * 0.6 )

        # Insert the removed class into the drift intervals 
        insert_images, insert_labels, inserts_labels_super = this.split_data(0,n_include_removed_1,
                                                                       this.removed_images,this.removed_labels, this.removed_labels_super)

        
        print("Insert removed labels to interval", len(insert_images))
        
        print("Before merge removed images into drift interval 1:", include_images_1.shape, include_labels_1.shape, include_labels_super_1.shape)
        
        include_images_1, include_labels_1, include_labels_super_1 = this.merge_data(include_images_1, insert_images,
                                                                                           include_labels_1, insert_labels,
                                                                                           include_labels_super_1, inserts_labels_super)

        
        print("after merge removed images into drift interval 1:", include_images_1.shape, include_labels_1.shape, include_labels_super_1.shape)
        
        include_images_1, include_labels_1, include_labels_super_1 = this.shuffle_data(include_images_1, include_labels_1, 
                                                                                                   include_labels_super_1)

        
        insert_images, insert_labels, insert_labels_super = this.split_data(n_include_removed_1,n_include_removed_2+n_include_removed_1,
                                                                       this.removed_images,this.removed_labels, this.removed_labels_super)
        
        print("Before merge removed images into drift interval 2:", include_images_2.shape, include_labels_2.shape, include_labels_super_2.shape)
        
        include_images_2, include_labels_2, include_labels_super_2 = this.merge_data(include_images_2,insert_images,
                                                                                           include_labels_2, insert_labels,
                                                                                           include_labels_super_2, insert_labels_super)
        
        print("After merge removed images into drift interval 2:", include_images_2.shape, include_labels_2.shape, include_labels_super_2.shape)

        last = n_include_removed_2+n_include_removed_1
        insert_images, insert_labels, insert_labels_super = this.split_data(last, last+n_include_removed_3,
                                                                       this.removed_images,this.removed_labels, this.removed_labels_super)
        print("Before merge removed images into drift interval 2:", include_images_3.shape, include_labels_3.shape, include_labels_super_3.shape)
        
        include_images_3, include_labels_3, include_labels_super_3 = this.merge_data(include_images_3,insert_images,
                                                                                           include_labels_3, insert_labels,
                                                                                           include_labels_super_3, insert_labels_super)
        
        print("After merge removed images into drift interval 2:", include_images_2.shape, include_labels_2.shape, include_labels_super_2.shape)
        print(f"Added a total of {n_include_removed_1 + n_include_removed_2+n_include_removed_3} from removed class")

        include_images_2, include_labels_2, include_labels_super_2 = this.shuffle_data( include_images_2, include_labels_2, 
                                                                                                   include_labels_super_2)

        include_images_3, include_labels_3, include_labels_super_3 = this.shuffle_data( include_images_3, include_labels_3, 
                                                                                                   include_labels_super_3)
        
        exlusion_set_images, exlusion_set_labels, exlusion_set_labels_super = this.shuffle_data( exlusion_set_images, exlusion_set_labels, 
                                                                                                   exlusion_set_labels_super)

        exlude_set_length = len(exlusion_set_labels)
        exclude_set_midpoint = int(exlude_set_length // 2)
        

        # Merge the intervals into a simulation set
        stream_images,stream_labels, stream_labels_super = this.merge_data(exlusion_set_images, include_images_1, exlusion_set_labels, 
                                                                           include_labels_1, exlusion_set_labels_super, include_labels_super_1)

        
        stream_images,stream_labels, stream_labels_super = this.merge_data(stream_images, include_images_2, stream_labels, 
                                                                           include_labels_2, stream_labels_super, include_labels_super_2)

        stream_images,stream_labels, stream_labels_super = this.merge_data(stream_images, include_images_3, stream_labels, 
                                                                           include_labels_3, stream_labels_super, include_labels_super_3)
        
        # Define the drift points by the length of each interval. 
        this.drift1 = len(exlusion_set_images)
        this.drift2 = this.drift1+len(include_images_1)
        this.drift3 = this.drift2+len(include_images_2)
        this.drift4 = this.drift3+len(include_images_3)
        print(this.drift1,this.drift2,this.drift3,this.drift4)
        this.simulation_images, this.simulation_labels, this.simulation_labels_super = stream_images, stream_labels, stream_labels_super
        print("simulation length after introducing drift: ", len(this.simulation_images))


