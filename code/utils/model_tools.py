import os
import json
from tensorflow.contrib.keras.python import keras 
from scipy import misc
from . import data_iterator
import numpy as np
import glob

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_architecture(model, arch_file_path):
    arch_data = model.to_json()
    with open(arch_file_path, 'w') as file:
        json.dump(arch_data, file)

def load_architecture(arch_file_path):
    if os.path.exists(arch_file_path):
        with open(arch_file_path, 'r') as file:
            arch_data = json.load(file)
            assert arch_data is not None, "Architecture data was empty"
        model = keras.models.model_from_json(arch_data)
        assert model is not None, "Null model returned from keras' model_from_json utility"
        return model
    else:
        raise ValueError('No architecture file found at {}'.format(arch_file_path))

def load_network(arch_file_path, weights_file_path):
    assert os.path.is_file(arch_file_path), "{} does not exist, or is not a file".format(arch_file_path)
    assert os.path.is_file(weights_file_path), "{} does not exist, or is not a file".format(weights_file_path)
    model = load_architecture(arch_file_path)
    model.load_weights(weights_file_path)
    return model
        
"""
def save_network(your_model, your_weight_filename):
    config_name = 'config' + '_' + your_weight_filename
    weight_path = os.path.join('..', 'data', 'weights', your_weight_filename)
    config_path = os.path.join('..', 'data', 'weights', config_name)
    your_model_json = your_model.to_json()
    
    with open(config_path, 'w') as file:
        json.dump(your_model_json, file)  
        
    your_model.save_weights(weight_path) 
        

def load_network(your_weight_filename):
    config_name = 'config' + '_' + your_weight_filename
    weight_path = os.path.join('..', 'data', 'weights', your_weight_filename)
    config_path = os.path.join('..', 'data', 'weights', config_name)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            json_string = json.load(file)  
            
        your_model = keras.models.model_from_json(json_string)
        
    else:
        raise ValueError('No config_yourmodel file found at {}'.format(config_path))
        
    if os.path.exists(weight_path):
        your_model.load_weights(weight_path)
        return your_model
    else:
        raise ValueError('No weight file found at {}'.format(weight_path))
"""

def write_predictions_grade_set(model, grading_dir_name, subset_name, out_folder_suffix):
    validation_path = os.path.join('..', 'data', 'masks', grading_dir_name, subset_name)
    file_names = sorted(glob.glob(os.path.join(validation_path, 'images', '*.jpeg')))

    output_path = os.path.join('..', 'data', 'inferences', subset_name + '_' + out_folder_suffix)
    make_dir_if_not_exist(output_path)
    image_shape = model.layers[0].output_shape[1]

    print ("# files being inferred:", len(file_names))

    for name in file_names:
        image = misc.imread(name)
        if image.shape[0] != image_shape:
             image = misc.imresize(image, (image_shape, image_shape, 3))
        image = data_iterator.standardize(image.astype(np.float32))
        pred = model.predict_on_batch(np.expand_dims(image, 0))
        base_name = os.path.basename(name).split('.')[0]
        base_name = base_name + '_prediction.png'
        misc.imsave(os.path.join(output_path, base_name), np.squeeze((pred * 255).astype(np.uint8)))

    print ("\t{} -> {}".format(validation_path, output_path))
    return validation_path, output_path
