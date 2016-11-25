import numpy as np

class DataSet(object):
    def __init__(self,
                 list_of_data,
                 list_of_labels,
                 dtype=np.float32):
        self._data = list(list_of_data)
        self._labels = list(list_of_labels)
        self._num_regions = len(list_of_data)
        
        self._num_examples = min([len(list_of_data[i]) for i in range(self._num_regions)])
        self._index_in_epoch = 0
        self._index_in_eval_epoch = 0
        self._positive_data = None
        self._negative_data = None
        self.permute_data()
        self.split_data()
        self._pos_index = 0
        self._neg_index = 0
        

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples
    
    def split_data(self):
        self._positive_data = []
        self._negative_data = []
        for i in range(self._num_regions):
            positive_index = np.where(self._labels[i] > 0)[0]
            negative_index = np.where(self._labels[i] == 0)[0]
            self._positive_data.append(self._data[i][positive_index])
            self._negative_data.append(self._data[i][negative_index])
        self._pos_num = min([self._positive_data[i].shape[0] for i in range(self._num_regions)])
        self._neg_num = min([self._negative_data[i].shape[0] for i in range(self._num_regions)])
            
    def permute_data(self):
        for i in range(self._num_regions):
            perm = np.arange(self._data[i].shape[0])
            np.random.shuffle(perm)
            self._data[i] = self._data[i][perm]
            self._labels[i] = self._labels[i][perm]

    def next_batch(self, batch_size):
        """
        This function will return a list of data array and a list labels array
        each array in the list represent the data/label batch for a particular
        region
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Shuffle the data
            self.permute_data()
            self.split_data()
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        
        return [self._data[i][start:end] for i in range(self._num_regions)],\
            [self._labels[i][start:end] for i in range(self._num_regions)]
        
    def next_balanced_batch(self, batch_size, neg_ratio = 1.0):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Shuffle the data
            self.permute_data()
            self.split_data()
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        positive_size = int(1.0 * batch_size / (neg_ratio + 1))
        negative_size = int(neg_ratio * batch_size / (neg_ratio + 1))
        
        pos_start = self._pos_index % self._pos_num
        neg_start = self._neg_index % self._neg_num
        self._pos_index = pos_start + positive_size
        self._neg_index = neg_start + negative_size
        
        all_data = [np.array(list(self._positive_data[i][pos_start:pos_start + positive_size]) + 
                             list(self._negative_data[i][neg_start:neg_start + negative_size]))
                    for i in range(self._num_regions)]
        all_label = [np.array([1] * self._positive_data[i][pos_start:pos_start + positive_size].shape[0] + 
                              [0] * self._negative_data[i][neg_start:neg_start + negative_size].shape[0])
                              for i in range(self._num_regions)]
        
        return all_data, all_label

    def next_eval_batch(self, batch_size):
        start = self._index_in_eval_epoch
        self._index_in_eval_epoch += batch_size
        if start >= NUM_EXAMPLES_PER_EPOCH_FOR_EVAL:
            self._index_in_eval_epoch = 0
            return None, None
        else:
            end = self._index_in_eval_epoch
            return [self._data[i][start:end] for i in range(self._num_regions)],\
                [self._labels[i][start:end] for i in range(self._num_regions)]
