

import os, sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import re

"""# **PRE-PROCESSING**

# **First folder with Labels**

1.   List item
2.   List item

**Reading the whole directory**
"""

path = '/input/spinal/spinal_Classification/spinal/'
dirs = os.listdir(path)
print('dirs:', dirs)

l_folders = [] # Initialize a list to hold the full paths of subdirectories

for item in dirs: # Loop through each item in the directory
    temp_dir = path + item
    if os.path.isdir(temp_dir): # Check if the current item is a directory
        subdirs = os.listdir(temp_dir) # List the contents of the subdirectory
        for subitem in subdirs: # Loop through each item in the subdirectory
            l_folders.append(temp_dir + "/" + subitem) # Append the full path of the subitem to the list
            print(subitem)

print(l_folders)

l_name_files = []
l_images = []

for item in l_folders:
    file_dir_temp = os.listdir(item)
#     print('file_dir_temp', file_dir_temp)

    for item_list in file_dir_temp:
        path_img = item + '/' + item_list
#         print(path_img)
        img = mpimg.imread(path_img, format = 'jpg')
        l_images.append(img)
        l_name_files.append(item_list)

len(l_images)

l_images[1344].shape


l_name_files[0]

"lavum" in l_name_files[0]

plt.imshow(l_images[1344])

l_images_2D = [item[:,:,0] for item in l_images] # selects all pixels in the first channel of the image. This reduces each 3D image (height x width x color) to a 2D image (height x width)
# This reduces the data complexity for further processing tasks that do not require full color information

plt.imshow(l_images_2D[1344])

len(l_images_2D)

l_images_2D[123].shape

l_images[0]

l_images_2D[0]

# Converting to array
a_images_2D = np.array(l_images_2D)

a_images_2D.shape

a_images_2D[0]

"""***Label the Data***"""

l_label = []

dict_count = {}
dict_count["fat"] = 0
dict_count["flavum"] = 0
dict_count["ligament"] = 0
dict_count["spinalcord"] = 0

for item in l_name_files:

    if "fat" in item:
        l_label.append("fat")
        dict_count["fat"] = dict_count["fat"] + 1
    elif "flavum" in item:
        l_label.append("flavum")
        dict_count["flavum"] = dict_count["flavum"] + 1
    elif "spinalcord" in item:
        l_label.append("spinal_cord")
        dict_count["spinalcord"] = dict_count["spinalcord"] + 1
    elif "ligament" in item:
        l_label.append("ligament")
        dict_count["ligament"] = dict_count["ligament"] + 1

dict_count

a_label = np.array(l_label)
a_label

"""***Get spinal Number***"""

l_spinal_num = []

r = re.compile(r'(?<=e)[0-9]+')

for item in l_name_files:

  res = r.search(item)
#     print(int(res.group(0)))
  l_spinal_num.append(int(res.group(0)))

l_spinal_num[0:10]

l_spinal_num[10397]

l_name_files[10397]

a_spinal_num = np.array(l_spinal_num)

len(a_spinal_num)

np.unique(a_spinal_num)

"""**Saving as Binaries**"""

from google.colab import drive
drive.mount('/content/drive')

with open('/content/drive/MyDrive/a_images_2D.npy', 'wb') as f:
  np.save(f, a_images_2D)

with open('/content/drive/MyDrive/a_label.npy', 'wb') as f:
  np.save(f, a_label)

with open('/content/drive/MyDrive/a_spinal_num.npy', 'wb') as f:
  np.save(f, a_spinal_num)

"""# **Second Folder without labels i.e spinal Space**"""

path = 'input/spinal/spinal_Classification/spinal_new_class/'

dirs = os.listdir(path) # List directories and files within the path
print(dirs)
list_folders = []

for item in dirs:
  temp_dir = os.path.join(path, item)
  print(temp_dir)

  if os.path.isdir(temp_dir):
    sub_dirs = os.listdir(temp_dir)
    print(sub_dirs)

    for subitem in sub_dirs:
      list_folders.append(os.path.join(temp_dir, subitem))

list_folders[0:10]

list_file_name = []
list_images = []

for folder in list_folders:
  file_in_folder = os.listdir(folder)
  print(file_in_folder)

  for sub_file in file_in_folder:
    sub_file_path = os.path.join(folder, sub_file)
    print(sub_file_path)

    # Read and store image alomng with its file_name
    img = mpimg.imread(sub_file_path)
    list_file_name.append(sub_file)
    list_images.append(img)

list_images[0:10]

list_file_name[0:50]

list_file_name[1889]

list_images_2D = [i[:,:,0] for i in list_images]

arr_images_2D = np.array(list_images_2D)

arr_images_2D[65]

len(arr_images_2D)

"""**spinal Number**"""

import re

lst_spinal_num = []

r = re.compile(r'(?<=e)[0-9]+')

for item in list_file_name:

  res = r.search(item)
  lst_spinal_num.append(int(res.group(0)))

arr_spinal_num = np.array(lst_spinal_num)

arr_spinal_num[:20]

np.unique(arr_spinal_num)

def count_elements(arr):
  count_dict = {} # Dictonary to store the count of each element
  for element in arr:
    count_dict[element] = count_dict.get(element, 0) + 1
  return count_dict

count_elements(sorted(arr_spinal_num))

"""**Count Distances**"""

lst_distance = []

for file in list_file_name:

  if "0.5mm" in file:
      lst_distance.append("0.5mm")
  elif "1.0mm" in file:
      lst_distance.append("1.0mm")
  elif "1.5mm" in file:
      lst_distance.append("1.5mm")
  elif "2.0mm" in file:
      lst_distance.append("2.0mm")
  elif "2.5mm" in file:
      lst_distance.append("2.5mm")

arr_distance = np.array(lst_distance)

arr_distance[:30]

# def filter_and_sample(distances, ep_num, num_samples):
#   # Filter based on distance and spinal number
#   mask = np.logical_and(np.isin(a_distance, distances))

np.random.seed(1696)

lst_empty_imgs = []
lst_spinal_empty_num = []

# Loop over each spinal number starting from 1 to 8
for epid_num in np.arange(1, 9):
  imgs_concat = [] # List to temporarily store images for concatenation for this spinal number
  spinal_num_concat = [] # List to temporarily store spinal numbers for concatenation

  # Loop over each distance category with its corresponding sample size
  for distance, sample_size in zip(["0.5mm", "1.0mm", ">=1.5mm"], [333, 333, 334]):
    if distance == ">=1.5mm":
      # Combine multiple conditions for selecting distances 1.5mm to 2.5mm
      condition = np.logical_and.reduce((
          arr_distance >= "1.5mm", # Include distances greater than or equal to 1.5mm
          arr_distance <= "2.5mm", # But less than or equal to 2.5mm
          arr_spinal_num == epid_num # spinal number matches the current loop
      )) # .reduce in np.logical_and.reduce serves to combine multiple logical checks into one operation
    else:
      # For specific distances (0.5mm and 1.0mm), match exact distance and spinal number
      condition = np.logical_and(
          arr_distance == distance, # Match the exact distance
          arr_spinal_num == epid_num # Match the exact spinal number
      )

    # Get indices of items that meet the condition, then randomly select the specified number without replacement
    indices = np.random.choice(np.where(condition)[0], sample_size, replace = False)

    # Extract images for the selected indices and add to the temporary list
    imgs_concat.append(arr_images_2D[indices])
    # Extract spinal numbers for the selected indices and add to the temporary list
    spinal_num_concat.append(arr_spinal_num[indices])

  # Concatenate all selected images for this spinal number and add to the main list
  lst_empty_imgs.append(np.concatenate(imgs_concat, axis = 0))
  # Concatenate all selected spinal numbers for this spinal number and add to the main list
  lst_spinal_empty_num.append(np.concatenate(spinal_num_concat, axis=0))

arr_images_emp = np.concatenate(lst_empty_imgs)
arr_images_emp.shape

arr_spinal_emp_num = np.concatenate(lst_spinal_empty_num)
arr_images_emp.shape

def count_elements(arr):
  count_dict = {} # Dictonary to store the count of each element
  for element in arr:
    count_dict[element] = count_dict.get(element, 0) + 1
  return count_dict

count_elements(sorted(arr_spinal_empty_num))

arr_spinal_emp_num.shape

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_images_emp.npy', 'wb') as f:
  np.save(f, arr_images_emp)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_spinal_emp_num.npy', 'wb') as f:
  np.save(f, arr_spinal_emp_num)

"""# **Concating both the Folders**

**Open files from first folder**
"""

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_images_2D.npy', 'rb') as f:
  arr_imgs_2d = np.load(f)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_spinal_num.npy', 'rb') as f:
  arr_spinal_num = np.load(f)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_label.npy', 'rb') as f:
  arr_label = np.load(f)

"""**Open files from second folder**"""

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_images_emp.npy', 'rb') as f:
  arr_imgs_emp_2d = np.load(f)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_spinal_emp_num.npy', 'rb') as f:
  arr_spinal_emp_num = np.load(f)

"""**Joining the Datasets**"""

len(arr_imgs_emp_2d)

emp_labels = ['emp' for i in range(len(arr_imgs_emp_2d))]

emp_labels[:20]

arr_emp_labels = np.array(emp_labels)

len(emp_labels)

# Concatenate images
arr_imgs_2d_concat = np.concatenate((arr_imgs_2d, arr_imgs_emp_2d))

# Concatenate spinal numbers
arr_spinal_num_concat = np.concatenate((arr_spinal_num, arr_spinal_emp_num))

arr_imgs_2d_concat.shape

arr_spinal_num_concat.shape

def count_elements(arr):
  count_dict = {} # Dictonary to store the count of each element
  for element in arr:
    count_dict[element] = count_dict.get(element, 0) + 1
  return count_dict

count_elements(sorted(arr_spinal_num_concat))

arr_label_concat = np.concatenate((arr_label, arr_emp_labels))

count_elements(arr_label_concat)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_imgs_2D_con.npy', 'wb') as f:
  np.save(f, arr_imgs_2D_concat)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_spinal_num_con.npy', 'wb') as f:
  np.save(f, arr_spinal_num_concat)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_label_con.npy', 'wb') as f:
  np.save(f, arr_label_concat)

"""# **Model Building**"""

from google.colab import drive
drive.mount('/content/drive')

import os, sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import re

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_imgs_2D_con.npy', 'rb') as f:
  arr_imgs_2d_con = np.load(f)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_spinal_num_con.npy', 'rb') as f:
  arr_spinal_num_con = np.load(f)

with open('/content/drive/MyDrive/ML_projects/P3-spinal/arr_label_con.npy', 'rb') as f:
  arr_label_con = np.load(f)

"""**Required Libraries**"""

import tensorflow as tf
from tensorflow import keras

import sklearn
from sklearn.model_selection import GroupKFold

import numpy as np

from time import perf_counter

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

np.unique(arr_label_con)

arr_label_con == 'ligament'

arr_label_con[arr_label_con ==  'ligament']

len(arr_label_con[arr_label_con ==  'ligament'])

arr_spinal_num = np.unique(arr_spinal_num_con)
arr_spinal_num

# Convert labels from strings to integers using a predefined dictionary for consistent processing across the model.
arr_label_num = np.copy(arr_label_con)
label_dict = {'fat': 0, 'ligament': 1, 'flavum': 2, 'emp': 3, 'spinal_cord': 4}

for key, value in label_dict.items():
  arr_label_num[arr_label_num == key] = value
arr_label_num = arr_label_num.astype(int)

keras.backend.clear_session()  # Clears any previous model from memory to avoid interference.
np.random.seed(74)  # Seed for reproducibility of results in NumPy operations.
tf.random.set_seed(74)  # Seed for reproducibility of results in TensorFlow operations.

arr_imgs_2d = arr_imgs_2d_con[..., np.newaxis] # Adds a new axis to image data to match the input requirements of neural networks.

arr_imgs_2d.shape

K_test = 1,2,3,4
K_val = 5,6,7,8

arr_selected_spinal_test = np.array([1,2,3,4])
arr_selected_spinal_val = np.array([5,6,7,8])

arr_spinal_num_con[21281]

# # Initialize GroupKFold to manage cross-validation splitting according to groups defined by spinal numbers.
# k = 3 # No.of folds
# gkf = GroupKFold(n_splits = 3)

# # Perform k-fold cross-validation. Each fold uses different training and testing sets determined by the group (spinal numbers).
# for fold, (train_index, test_index) in enumerate(gkf.split(arr_imgs_2d, arr_label_num, groups = arr_spinal_n)):
#   print(f"Training fold {fold + 1}/k")

#   X_train, X_test = arr_imgs_2d[train_index], arr_imgs_2d[test_index] # Splits the image data into training and testing sets based on indices.
#   y_train, y_test = arr_label_num[train_index], arr_label_num[test_index] # Splits the labels into training and testing sets based on indices.

#   mean_train = np.mean(X_train) # Calculate the mean of training data for normalization.
#   X_train = X_train - mean_train # Normalize training data by subtracting the mean.
#   X_test = X_test - mean_train # Normalize training data by subtracting the mean (why training data mean? - to maintain consistency in data processing).

#   # Model setup using a ResNet50 architecture with modifications for the top layer to fit the specific classification task.
#   base_model = keras.applications.ResNet50(include_top = False, weights = None, input_shape = (241, 181, 1))
#   avg = keras.layers.GlobalAveragePooling2D()(base_model.output) # Applies global average pooling to the output of the base model.
#   output = keras.layers.Dense(5, activation = 'softmax')(avg) # Adds a dense output layer with 5 units for classification with softmax activation.
#   model_resnet = keras.models.Model(inputs = base_model.input, outputs = output) # Constructs the complete model.

#   optimizer = keras.optimizers.Adam(learning_rate = 0.001) # Using Adam optimizer with the default learning rate
#   model_resnet.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]) # Compiles the model with loss function and optimizer.

#   t1_start = perf_counter()

#   early_stopping_cb = keras.callbacks.EarlyStopping(patience= 2, restore_best_weights=True)  # Setup early stopping for efficient training.
#   model_resnet.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), callbacks=[early_stopping_cb])

#   t1_stop = perf_counter()
#   time_lapse = t1_stop-t1_start

#   # Save the model and results for each fold
#   model_resnet.save(f"/path/to/save/model_spinal_fold_{fold+1}.h5")

#   print( "Elapsed time during the whole program in seconds for current fold is :", time_lapse)
#   print('////////////////////////////////////////////////////////////////////////////////////')

print(f"Size of image data array: {len(arr_imgs_2d)}")
print(f"Size of label array: {len(arr_label_num)}")
print(f"Size of group array: {len(arr_spinal_num)}")

# for index in np.arange(1,11,1):
for index in arr_selected_spinal_test:

    print("Spinal testing: " + str(index) )

    arr_spinal_num_val = np.delete(arr_selected_spinal_val, np.where( arr_selected_spinal_val == index))
    print("a_spinal_num_val = ", arr_spinal_num_val)

    bool_spinal_num = arr_spinal_num_con != index
    a_images_1D_7_spinals = arr_imgs_2d[bool_spinal_num]
    a_label_num_7_spinals = arr_label_num[bool_spinal_num]
    a_spinal_num_7_spinals = arr_spinal_num_con[bool_spinal_num]

    print(len(a_images_1D_7_spinals))
    print(len(a_label_num_7_spinals))

    X_cv = a_images_1D_7_spinals
    y_cv = a_label_num_7_spinals

    n_epochs = 5
    # n_epochs = 1

    batch_size = 8

    # k fold validation  max:7 fold

    for index_val in arr_spinal_num_val:

        print("spinal_val: " + str(index_val) )

        bool_val_spinal = ( a_spinal_num_7_spinals == index_val )
        print('bool_val_spinal', bool_val_spinal)
        bool_train_spinal = ~bool_val_spinal
        print('bool_train_spinal', bool_train_spinal)

        X_train_raw, X_val_raw = X_cv[bool_train_spinal], X_cv[bool_val_spinal]
        y_train, y_val = y_cv[bool_train_spinal], y_cv[bool_val_spinal]

        # # Check if training or validation sets are empty
        # if X_train_raw.size == 0 or X_val_raw.size == 0:
        #     print(f"Empty training or validation set for spinal_val {index_val}. Skipping.")
        #     continue

        # Preprocessing data
        # Substracting the mean pixel
        # (This is done normally per channel)
        mean_train_raw = np.mean(X_train_raw[:100])
        print("mean_train_raw = ", mean_train_raw)

        X_train = X_train_raw[:100] - mean_train_raw
        X_val = X_val_raw[:100] - mean_train_raw

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        #ResNet50

        base_model_empty = keras.applications.resnet50.ResNet50( include_top=False,
                                              weights=None,
                                              input_tensor=None,
                                              input_shape=(241,181,1),
                                              pooling=None)

        n_classes=5

        avg = keras.layers.GlobalAveragePooling2D()(base_model_empty.output)
        output = keras.layers.Dense(n_classes, activation="softmax")(avg)
        model_resnet50 = keras.models.Model(inputs=base_model_empty.input, outputs=output)

        # Using an exponential decay learning rate schedule
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                          restore_best_weights=True)

        # time_cb = TimingCallback()

        model_resnet50.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                      metrics=["accuracy"])

        t1_start = perf_counter()

        history = model_resnet50.fit(X_train[:100], y_train[:100],
                               batch_size=batch_size,
                               validation_data=(X_val[:70], y_val[:70]),
                               epochs=n_epochs,
                               callbacks=[early_stopping_cb])

        # RESNET 50

        t1_stop = perf_counter()
        time_lapse = t1_stop-t1_start

        model_resnet50.save("model_outer_K%s_outer_k%s_val.h5"%(index, index_val))

        print( "Elapsed time during the whole program in seconds for K%s_outer_k%s_val: "%(index, index_val), time_lapse)

