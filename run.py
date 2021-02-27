# run.py

import argparse
import yaml

import src
from importlib import import_module

def import_from_path(path_to_module, obj_name = None):
    """
    Import an object from a module based on the filepath of
    the module and the string name of the object.
    If obj_name is None, return the module instead.
    """
    module_name = path_to_module.replace("/",".").strip(".py")
    module = import_module(module_name)
    if obj_name == None:
        return module
    obj = getattr(module, obj_name)
    return obj

if __name__ == "__main__":
    
    #GET THE ARGUMENTS
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("-c", "--config", help = "File path to the config file")
    parser.add_argument("-l", "--language", help = "Language on which we perform aspect extraction")
    parser.add_argument("-k", help="Number of aspects")
    parser.add_argument("-s", "--sentence", required=True)
    args = parser.parse_args()

    #LOAD CONFIG FILE
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
        
    #LOAD LANGUAGE        
    w2v = src.datasets.read_english_w2v(lim=10000)
    train_texts, train_labels = src.datasets.read_imdb()
        
    sentence = config['sentences']['sentence%s'%args.sentence]
    
    #LOAD DETECTOR
    detector_params = config["model"]["parameters"]
    
    if args.k != None:
        print(type(args.k))
        detector_params['k'] = args.k
    
    detector = import_from_path(config["model"]["filepath"],
                                config["model"]["class"])(w2v, **detector_params)
    
    #detector = kmeans.KMeansAspectDetector(w2v, k=args.k)
    res = detector.predict_sentence(sentence)
    
    print(res)