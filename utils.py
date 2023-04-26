### Utils
import h5py
import os
from PIL import Image


import matplotlib.pyplot as plt
import numpy as np
import pickle


def load_it_data(path_to_data):
    """ Load IT data

    Args:
        path_to_data (str): Path to the data

    Returns:
        np.array (x6): Stimulus train/val/test; objects list train/val/test; spikes train/val
    """

    datafile = h5py.File(os.path.join(path_to_data,'IT_data.h5'), 'r')

    stimulus_train = datafile['stimulus_train'][()]
    spikes_train = datafile['spikes_train'][()]
    objects_train = datafile['object_train'][()]
    
    stimulus_val = datafile['stimulus_val'][()]
    spikes_val = datafile['spikes_val'][()]
    objects_val = datafile['object_val'][()]
    
    stimulus_test = datafile['stimulus_test'][()]
    objects_test = datafile['object_test'][()]
    

    ### Decode back object type to latin
    objects_train = [obj_tmp.decode("latin-1") for obj_tmp in objects_train]
    objects_val = [obj_tmp.decode("latin-1") for obj_tmp in objects_val]
    objects_test = [obj_tmp.decode("latin-1") for obj_tmp in objects_test]

    return stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val


def visualize_img(stimulus,objects,stim_idx):
    """Visualize image given the stimulus and corresponding index and the object name.

    Args:
        stimulus (array of float): Stimulus containing all the images
        objects (list of str): Object list containing all the names
        stim_idx (int): Index of the stimulus to plot
    """    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    img_tmp = np.transpose(stimulus[stim_idx],[1,2,0])

    ### Go back from normalization
    img_tmp = (img_tmp*normalize_std + normalize_mean) * 255

    plt.figure()
    plt.imshow(img_tmp.astype(np.uint8),cmap='gray')
    plt.title(str(objects[stim_idx]))
    plt.show()
    return


def visualize_imgs(stimulus, objects, stim_idxs):
    """Visualize images given the stimulus and corresponding indices and the object names.
    Similar to visualize_img, but vizualize more than one image at once. 

    Args:
        stimulus (array of float): Stimulus containing all the images
        objects (list of str): Object list containing all the names
        stim_idxs (list of int): List of indices of the stimuli to plot
    """    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    n_imgs = len(stim_idxs)

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))

    for i, stim_idx in enumerate(stim_idxs):
        row = i // 5
        col = i % 5
        img_tmp = np.transpose(stimulus[stim_idx],[1,2,0])
        img_tmp = (img_tmp*normalize_std + normalize_mean) * 255

        axes[row, col].imshow(img_tmp.astype(np.uint8), cmap='gray')
        axes[row, col].set_title(str(objects[stim_idx]))

    plt.tight_layout()
    plt.show()
    return

def create_mask(objects, object_name):
    '''
    Create a mask for the object class of interest.
    
    Args:
        objects (list of str): list containing all the classes of the images 
        object_name (str): name of the class of interest
    '''
    mask = [True if object_name in s else False for s in objects]
    mask_pos = [t.tolist() for t in np.where(mask)][0]
    return mask_pos

def print_category(stimulus, objects, mask_pos):
    """
    Print the number of images in the stimulus that contain the specified objects,
    as well as the number of unique objects in the stimulus and their indices.
    Visualize the images that contain the unique objects.

    Args:
        stimulus (array of float): Array of images to analyze
        objects (list of str): List of object names corresponding to each image in the stimulus
        mask_pos (list of int): List of indices specifying which objects to consider

    Returns:
        None
    """
    objs = [objects[i] for i in mask_pos]
    print('the number of stimulus with {} object is:'.format(objs[0]), len(objs))
    voc = list(set(objs))
    first_appearences = {}
    for i, term in enumerate(objects):
        if term not in first_appearences and term in voc:
            first_appearences[term] = i
    first_appearences = dict(sorted(first_appearences.items()))
    idxs = np.array(list(first_appearences.values()))
    print('the number of different face objects (different orientation of faces) is:', len(idxs))
    visualize_imgs(stimulus, objects, idxs)
    return 


def plot_average_firing_rate_for_category(spikes, mask_pos, neuron_idx, category = None):
    """
    Plot the average firing rate of a given neuron for a specified category (or all categories),
    over a specified time window (70-170 ms).

    Args:
        spikes (array of float): Array of spike counts, with dimensions (num_stimuli, num_neurons)
        mask_pos (list of int): List of indices specifying which stimuli belong to the category of interest
        neuron_idx (int): Index of the neuron to plot firing rates for
        category (str, optional): Category name to include in the plot title. If None, all categories are plotted.

    Returns:
        None
    """
    
    plt.figure()
    plt.title('Average firing rate for neuron {} and category {} (70-170 ms)'.format(neuron_idx, category))
    plt.plot(spikes[mask_pos,neuron_idx])
    return

def plot_average_firing_rate_for_all_categories_merged(objects, spikes, neuron_idx):
    
    """Plots the average firing rate of a single neuron for all categories merged.

    Args:
        objects (list of str): Object list containing all the object names
        spikes (array of float): Spike count data with shape (n_stimuli, n_neurons)
        neuron_idx (int): Index of the neuron to plot

    Returns:
        None
    """
    
    categories = ['Animals', 'Boats', 'Cars', 'Chairs', 'Faces', 'Fruits', 'Planes', 'Tables']
    
    face_pos = create_mask(objects, 'face')
    car_pos = create_mask(objects, 'car')
    chair_pos = create_mask(objects, 'chair')
    airplane_pos = create_mask(objects, 'airplane')
    ship_pos = create_mask(objects, 'ship')
    table_pos = create_mask(objects, 'table')

    # create mask for fruit objetcs. The problem here is that fruits are not all called 'fruit'
    # so we need to create a mask for each fruit and then combine them
    fruits = ['apple', 'apricot', 'peach', 'pear', 'raspberry', 'strawberry', 'walnut', 'watermelon']
    fruit_pos = []
    for fruit in fruits:
        fruit_pos.append(create_mask(objects, fruit))
    fruit_pos = np.concatenate(fruit_pos)

    # same for animals
    animals = ['bear', 'cow', 'dog', 'elephant', 'gorilla', 'hedgehog', 'lioness', 'turtle']
    animal_pos = []
    for animal in animals:
        animal_pos.append(create_mask(objects, animal))
    animal_pos = np.concatenate(animal_pos)

    #create a single array
    masks_no_animal = [ship_pos, car_pos, chair_pos, face_pos, fruit_pos, airplane_pos, table_pos]
    merged_by_category = animal_pos
    v_lines = [len(merged_by_category)]
    for i, mask in enumerate(masks_no_animal):
        merged_by_category = np.append(merged_by_category, mask)
        v_lines.append(len(merged_by_category))


    # now let's plot the average firing rate for all categories for neuron_idx
    fig, ax = plt.subplots(figsize = (15,5))
    ax.plot(spikes[merged_by_category,neuron_idx])

    # Add background colors to the plot based on the category
    ax.axvspan(0, v_lines[0], facecolor='grey', alpha=0.3)
    for i in range(1, len(v_lines)-1, 2):
        ax.axvspan(v_lines[i], v_lines[i+1], facecolor='grey', alpha=0.3)

    # Set the x-tick positions
    tick_positions = [v_lines[0]/2, 
                    (v_lines[0]+v_lines[1])/2, 
                    (v_lines[1]+v_lines[2])/2, 
                    (v_lines[2]+v_lines[3])/2, 
                    (v_lines[3]+v_lines[4])/2, 
                    (v_lines[4]+v_lines[5])/2, 
                    (v_lines[5]+v_lines[6])/2,
                    (v_lines[6]+v_lines[7])/2]
    ax.set_xticks(tick_positions)

    # Set the x-tick labels
    ax.set_xticklabels(['Animals', 'Boats', 'Cars', 'Chairs', 'Faces', 'Fruits', 'Planes', 'Tables'])
    plt.title('Average firing rate for neuron {} (70-170 ms)'.format(neuron_idx))
    plt.show()
    return 


def transform_to_8_classes(objects):
    '''
    Transform the input list of objects into one of 8 classes: animals, boats, cars, chairs, faces, fruits, planes, or tables.
    Args:
        objects (list of str): A list of object names, where each name corresponds to one of the 64 classes.  
    Returns:
        transformed_objects (list of str): A list of object names, where each name corresponds to one of the 8 classes.
    '''

    fruits = ['apple', 'apricot', 'peach', 'pear', 'raspberry', 'strawberry', 'walnut', 'watermelon']
    animals = ['bear', 'cow', 'dog', 'elephant', 'gorilla', 'hedgehog', 'lioness', 'turtle']
    objects = ['face' if 'face' in s else s for s in objects]
    objects = ['car' if 'car' in s else s for s in objects]
    objects = ['chair' if 'chair' in s else s for s in objects]
    objects = ['airplane' if 'airplane' in s else s for s in objects]
    objects = ['table' if 'table' in s else s for s in objects]
    objects = ['fruit' if s in fruits else s for s in objects]
    objects = ['animal' if s in animals else s for s in objects]
    objects = ['ship' if 'ship' in s else s for s in objects]
    return objects

def transform_classes_to_int(objects):
    """
    Transforms a list of object class names into a list of integers using a pre-defined mapping.
    
    Args:
        objects (list of str): List of object class names.

    Returns:
        objects_int (list of int): List of object class integers based on the pre-defined mapping.
    """
    # define a dictionary that maps the classes to integers
    classes = {'animal':0, 'ship':1, 'car':2, 'chair':3, 'face':4, 'fruit':5, 'airplane':6, 'table':7}
    # create a list of integers
    objects_int = [classes[object] for object in objects]  
    
    return objects_int


def save_image(stimulus, objects, stim_idx, output_path):
    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    img_tmp = np.transpose(stimulus[stim_idx],[1,2,0])

    ### Go back from normalization
    img_tmp = (img_tmp*normalize_std + normalize_mean) * 255

    plt.figure()
    plt.imshow(img_tmp.astype(np.uint8))
    file_name = os.path.join(output_path, f'image_{stim_idx}.png')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
    
def save_images_from_h5(file_path, output_path):
    """Create image folder from IT_data.h5 file.

    Args:
        path_to_data (str): Path to the IT_data.h5 file.
        output_folder (str): Path to the output folder where the images will be saved.

    Returns:
        None
    """
    
    # Create output subfolders for each stimulus set
    os.makedirs(os.path.join(output_path, 'stimulus-train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'stimulus-val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'stimulus-test'), exist_ok=True)        
        
    datafile = h5py.File(os.path.join(file_path,'IT_data.h5'), 'r')

    stimulus_train = datafile['stimulus_train'][()]
    object_train = datafile['object_train'][()]
    stimulus_val = datafile['stimulus_val'][()]
    object_val = datafile['object_val'][()]
    stimulus_test = datafile['stimulus_test'][()]
    object_test = datafile['object_test'][()]

    output_dir = os.path.join(output_path, 'stimulus-train')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx in range(stimulus_train.shape[0]):
        save_image(stimulus_train, object_train, idx , output_dir)
    
    
    output_dir = os.path.join(output_path, 'stimulus-val')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for idx in range(stimulus_val.shape[0]):
        save_image(stimulus_val, object_val, idx , output_dir)
        
        
    output_dir = os.path.join(output_path, 'stimulus-test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for idx in range(stimulus_test.shape[0]):
        save_image(stimulus_test, object_test, idx , output_dir)



def load_pickle_dict(file_path):
    """ Load the pkl dictionary into memory
    
    Args: 
        file_path (string): file path were the pickle dictionary is stored
    
    Returns:
        dictionary of values
    """
    
    with open(file_path, "rb") as f:
        file = pickle.load(f)
    return file


def compute_corr(true_vals, preds):
    """ Returns the overall correlation between real and predicted values in case the 
    number of neurons under study is 168.
    
    Args:
        true_vals (array of float):  true values 
        output_folder (array of float): pred values

    Returns:
        overall correlation coefficient
    """
    
    return np.mean(np.diag(np.corrcoef(true_vals, preds, rowvar = False)[:168, 168:]))



def load_test_data(path_to_data):
    """ Load IT data

    Args:
        path_to_data (str): Path to the data

    Returns:
        np.array (x6): Stimulus train/val/test; objects list train/val/test; spikes train/val
    """
    
    datafile = h5py.File(os.path.join(path_to_data,'IT_data.h5'), 'r')
    
    stimulus_test = datafile['stimulus_test'][()]
    spikes_test = datafile['spikes_test'][()]
    objects_test = [obj_tmp.decode("latin-1") for obj_tmp in objects_test]

    return stimulus_test, spikes_test, objects_test

