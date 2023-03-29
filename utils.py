### Utils
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np


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

    Args:
        stimulus (array of float): Stimulus containing all the images
        objects (list of str): Object list containing all the names
        stim_idxs (list of int): List of indices of the stimuli to plot
    """    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    n_imgs = len(stim_idxs)
    #n_cols = int(np.ceil(n_imgs/2))
    #n_rows = 2

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
    mask = [True if object_name in s else False for s in objects]
    mask_pos = [t.tolist() for t in np.where(mask)][0]
    return mask_pos

def print_category(stimulus, objects, mask_pos):
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
    plt.figure()
    plt.title('Average firing rate for neuron {} and category {} (70-170 ms)'.format(neuron_idx, category))
    plt.plot(spikes[mask_pos,neuron_idx])
    return



def plot_average_firing_rate_for_all_categories_merged(objects, spikes, neuron_idx):
    
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
