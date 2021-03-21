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
    parser.add_argument("-c", "--config", help = "File path to the config file", default="config/kmeans.yml")
    parser.add_argument('-e', "--embedding", 
                        help = "Which word embedding to use. 'w2v' or 'BERT'", 
                        default="w2v",
                        choices=['w2v', 'BERT'])
    parser.add_argument("-l", "--language", 
                        help = "Language on which we perform aspect extraction : 'fr' or 'eng'.", 
                        default="eng",
                        choices=['eng', 'fr'])
    parser.add_argument("-k", help="Number of aspects", default=5)
    parser.add_argument("-s", "--sentence", default=0)
    parser.add_argument("-t", "--train", action='store_true')
    args = parser.parse_args()

    #LOAD CONFIG FILE
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
        
    #LOAD LANGUAGE  
    language = args.language

    #WORD EMBEDDING    
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
        #Language is fr
        if emb == 'w2v':
            print("Selected Embedding : Word2Vec")
            wemb = src.datasets.read_french_w2v(**config['embedding']['w2v'])
            train_texts, train_labels = src.datasets.read_allocine()
        elif emb == 'BERT':
            print("Selected Embedding : BERT")
            raise Exception("Not Implemented Yet.")

    #LOAD DETECTOR
    detector_params = config["model"]["parameters"]
    
    if args.k != None:
        detector_params['k'] = int(args.k)
    if language == 'fr' and ('abae' in args.config):
        detector_params['dim_emb'] = 200
    
    detector = import_from_path(config["model"]["filepath"],
                                config["model"]["class"])(wemb, dataset = train_texts, **detector_params)
    
    if args.train:
        detector.train()
        res = detector.predict_aspect()
        print(res)
#     else :
#         sentence = config[language]['sentences']['sentence%s'%args.sentence]
#         print(f"Applying model to\n{sentence}")
#         res = detector.predict_sentence(sentence)
#         print(f"result:\n{res}")
    else:
        res = detector.predict_aspect()
        print(res)
        
        
        
        
        
        
    
    