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
    parser.add_argument('-e', "--embedding", help = "Which word embedding to use. 'w2v' or 'BERT' ")
    parser.add_argument("-l", "--language", help = "Language on which we perform aspect extraction : 'fr' or 'eng' ")
    parser.add_argument("-k", help="Number of aspects")
    parser.add_argument("-s", "--sentence", required=True)
    args = parser.parse_args()

    #LOAD CONFIG FILE
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
        
    #LOAD LANGUAGE
    language = 'eng'
    
    if args.language != None:
        if args.language == 'fr':
            print("Selected Language : French")
            language = 'fr'
            raise Exception("Not Implemented Yet.")
        elif args.language == 'eng':
            print("Selected Language : English")
        else:
            raise NameError("There is no '{}' language. Try 'en' or 'fr'.".format(args.language))
    else:
        print("Selected Language : English")
        
    #WORD EMBEDDING
    
    emb = 'w2v'
    
    if args.embedding != None:
        emb = args.embedding
        
    if language == 'eng':
        if emb == 'w2v':
            print("Selected Embedding : Word2Vec")
            wemb = src.datasets.read_english_w2v(**config['embedding']['w2v'])
            train_texts, train_labels = src.datasets.read_imdb()
        elif emb == 'BERT':
            print("Selected Embedding : BERT")
            raise Exception("Not Implemented Yet.")
        else:
            raise NameError("There is no '{}' language. Try 'en' or 'fr'.".format(args.language))
            
    elif language == 'fr':
        raise
        
    #LOAD TEXT TO TEST ON 
        
    sentence = config['sentences']['sentence%s'%args.sentence]
    
    #LOAD DETECTOR
    detector_params = config["model"]["parameters"]
    
    if args.k != None:
        detector_params['k'] = int(args.k)
    
    detector = import_from_path(config["model"]["filepath"],
                                config["model"]["class"])(wemb, **detector_params)
    
    res = detector.predict_sentence(sentence)
    
    print(res)