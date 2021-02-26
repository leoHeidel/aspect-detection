# run.py

import argparse
import yaml

import src
import kmeans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("-c", "--config", help = "File path to the config file")
    parser.add_argument("-k", help="Number of aspects")
    parser.add_argument("-s", "--sentence", required=True)
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
            
    w2v = src.datasets.read_english_w2v(lim=10000)
    train_texts, train_labels = src.datasets.read_imdb()
        
    sentence = config['sentences']['sentence%s'%args.sentence]
    
    detector = kmeans.KMeansAspectDetector(w2v, k=args.k)
    res = detector.predict_sentence(sentence)
    
    print(res)