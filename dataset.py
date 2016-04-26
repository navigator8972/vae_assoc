import numpy as np

class DataSets(object):
    pass

class DataSet(object):
  def __init__(self, data, labels=None):
      if labels is not None:
          #check consistency
          assert data.shape[0]==labels.shape[0], (
              'data.shape: %s labels.shape: %s' % (data.shape,
                                                    labels.shape))
      else:
          #goahead
          self._num_examples = data.shape[0]
      self._data = data
      self._labels = labels
      self._epochs_completed = 0
      self._index_in_epoch = 0
      return

  def next_batch(self, batch_size):
      """Return the next `batch_size` examples from this data set."""
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      if self._index_in_epoch > self._num_examples:
         # Finished epoch
         self._epochs_completed += 1
         # Shuffle the data
         perm = np.arange(self._num_examples)
         np.random.shuffle(perm)
         self._data = self._data[perm]
         if self._labels is not None:
             self._labels = self._labels[perm]
         # Start next epoch
         start = 0
         self._index_in_epoch = batch_size
         assert batch_size <= self._num_examples
      end = self._index_in_epoch
      if self._labels is not None:
          return self._data[start:end], self._labels[start:end]
      else:
          return self._data[start:end], None

def construct_datasets(data, labels=None, shuffle=True, validation_ratio=.1, test_ratio=.1):

    data_sets = DataSets()
    if shuffle:
        perm = np.arange(data.shape[0])
        np.random.shuffle(perm)
        data_shuffled = data[perm]
        if labels is not None:
            labels_shuffled = labels[perm]
    else:
        data_shuffled = data
        labels_shuffled = labels

    test_start_idx = int((1-test_ratio)*data_shuffled.shape[0])
    validation_start_idx = int((1-validation_ratio-test_ratio)*data_shuffled.shape[0])
    if labels is not None:
        assert data_shuffled.shape[0] == labels_shuffled.shape[0], (
            'data.shape: %s labels.shape: %s' % (data.shape,
                                                  labels.shape))
        data_sets.train = DataSet(data_shuffled[:validation_start_idx, :], labels_shuffled[:validation_start_idx, :])
        data_sets.validation = DataSet(data_shuffled[validation_start_idx:test_start_idx, :], labels_shuffled[validation_start_idx, test_start_idx, :])
        data_sets.test = DataSet(data_shuffled[test_start_idx:, :], labels_shuffled[test_start_idx:, :])
    else:
        data_sets.train = DataSet(data_shuffled[:validation_start_idx, :])
        data_sets.validation = DataSet(data_shuffled[validation_start_idx:test_start_idx, :])
        data_sets.test = DataSet(data_shuffled[test_start_idx:, :])
        
    return data_sets
