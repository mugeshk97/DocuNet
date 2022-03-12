import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

class DataLoader:
    """
    The data loader class is used to load data from a directory and generate batches of data.
    """
    def __init__(self, shape, batch_size, scale = True):
        """
        Initializes the data loader class.
        Args:
            shape: shape of the image
            batch_size: size of the batch
            scale: whether to scale the image to the range [0, 1]
        """
        self.indices = {}
        self.__file_names__ = []
        self.__labels__ = []
        self.__shape__ = shape
        self.__batch_size__ = batch_size
        self.__scale__ = scale
    
    def load_from_directory(self, folder_directory):
        """
        Loads data from a given directory.
        Args:
            folder_directory: directory of the data
        Returns:
            None
        """
        self.num_files = 0
        classes = os.listdir(folder_directory)
        for i in range(len(classes)):
            sub_dir = os.path.join(folder_directory, classes[i])
            filename = os.listdir(sub_dir)
            for file_ in filename:
                file_path = os.path.join(sub_dir, file_)
                self.__file_names__.append(file_path)
                self.num_files += 1
                self.__labels__.append(i)
            self.indices[i] = classes[i]
        print(f"Loaded {self.num_files} files of {len(classes)} classes")
        return None

    def load_from_report(self, file_name):
        """
        Loads data from a given report file.
        Args:
            file_name: name of the report file
        Returns:
            None
        """
        df = pd.read_json(file_name)
        self.num_files = len(df)
        for i in range(len(df)):
            self.__file_names__.append(df.iloc[i]["filename"])
            data = {}
            for col in df.columns:
                if col != 'filename' and col != 'instruction':
                    if df.iloc[i][col] != "":
                        data[col] = 1
                    else:
                        data[col] = 0
                    self.__labels__.append(data)
        print(f"Loaded {self.num_files} files of {len(self.indices)} classes")
        return None

    def data_generator(self, shape = (360,360), batch_size = 4, scale = True):
        """
        generator for the data loader class.
        Args:
            shape: shape of the image
            batch_size: size of the batch
            scale: whether to scale the image to the range [0, 1]

        Returns:
            batch of images and labels
        """
        self.__shape__ = shape
        self.__batch_size__ = batch_size
        self.__scale__ = scale

        if len(self.__file_names__) == 0:
            raise Exception("No data loaded")
        else:
            while True:
                batch_y_train = {}
                batch_x = np.zeros(shape = (self.__batch_size__, self.__shape__[0], self.__shape__[1], 1))
                batch_y = np.zeros(shape = (self.__batch_size__))
                for i in range(self.__batch_size__):
                    index = np.random.randint(0, len(self.__file_names__))
                    img = cv2.imread(self.__file_names__[index])
                    img = cv2.resize(img, self.__shape__)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if self.__scale__:
                        img = img / 255.0 
                    img = np.expand_dims(img, axis = -1) # add a channel dimension
                    batch_x[i] = img
                    batch_y[i] = self.__labels__[index]
                y = tf.one_hot(batch_y, depth = len(self.indices))
                for i in self.indices.keys():
                    batch_y_train[self.indices[i]] = y[:, i]
                yield batch_x, batch_y_train