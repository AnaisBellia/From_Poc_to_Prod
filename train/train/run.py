import os
import json
import argparse
import time
import logging

import keras.layers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from keras import Sequential
from keras.layers import Dense, Input  # tensorflow.keras.layers

from preprocessing.preprocessing.embeddings import embed
from preprocessing.preprocessing.utils import LocalTextCategorizationDataset

logger = logging.getLogger(__name__)


def train(dataset_path, train_conf, model_path, add_timestamp):
    """
    :param dataset_path: path to a CSV file containing the text samples in the format
            (post_id 	tag_name 	tag_id 	tag_position 	title)
    :param train_conf: dictionary containing training parameters, example :
            {
                batch_size: 32 # train_conf[0]
                epochs: 1
                dense_dim: 64
                min_samples_per_label: 10  # train_conf[3]
                verbose: 1
            }
    :param model_path: path to folder where training artefacts will be persisted
    :param add_timestamp: boolean to create artefacts in a sub folder with name equal to execution timestamp
    """

    # if add_timestamp then add sub folder with name equal to execution timestamp '%Y-%m-%d-%H-%M-%S'
    if add_timestamp:
        artefacts_path = os.path.join(model_path, time.strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        artefacts_path = model_path

    # TODO: CODE HERE
    # instantiate a LocalTextCategorizationDataset, use embed method from preprocessing module for preprocess_text param
    # use train_conf for other needed params
    #dataset = LocalTextCategorizationDataset(dataset_path,batch_size=train_conf[0], min_samples_per_label=train_conf[3], preprocess_text=embed())
    dataset = LocalTextCategorizationDataset(dataset_path, batch_size=train_conf['batch_size'],
                                             min_samples_per_label=train_conf['min_samples_per_label'],
                                             preprocess_text=embed) #
    logger.info(dataset)

    # TODO: CODE HERE
    # instantiate a sequential keras model
    # add a dense layer with relu activation
    # add an output layer (multiclass classification problem)

    input_shape = (dataset.get_train_batch()[0].shape[-1],)

    #input_shape = dataset.get_train_batch()[0].shape(-1)
    model = keras.Sequential(
        [
    Input(shape=input_shape), # -1 dernier elemnt de batch, car taille de vec
            # 0 car recupere x title le premier element, on veut la taille de l'embedding
            # x(batch size, feature size), 0 : batch size, 1: feature size
            # on veut le feature size qui represente le nb de colonnes
            # donc on prend x[0].shape(-1)
    Dense(train_conf['dense_dim'], activation='relu', input_shape=input_shape),
    Dense(dataset.get_num_labels(), activation='sigmoid')
        ]
    )
    # TODO: CODE HERE
    # model fit using data sequences
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_history = model.fit(dataset.get_train_sequence(),
            epochs=train_conf['epochs'],
            verbose=train_conf["verbose"],
            validation_data=dataset.get_test_sequence(),
            ## à enlever si bug
            batch_size=train_conf['batch_size'],
        )

    # scores
    scores = model.evaluate_generator(dataset.get_test_sequence(), verbose=0)
    print("scores : " , scores)

    logger.info("Test Accuracy: {:.2f}".format(scores[1] * 100))

    # TODO: CODE HERE
    # create folder artefacts_path
    try:
        os.makedirs(artefacts_path)
    except:
        pass

    # TODO: CODE HERE
    # save model in artefacts folder, name model.h5
    model.save(f"{artefacts_path}/model.h5")

    # TODO: CODE HERE
    # save train_conf used in artefacts_path/params.json
    with open(f'{artefacts_path}/params.json', 'w') as f:
        json.dump(train_conf, f)

    # TODO: CODE HERE
    # save labels index in artefacts_path/labels_index.json
    with open(f'{artefacts_path}/labels_index.json', 'w') as f:
        labels_index = dataset.get_index_to_label_map()
        json.dump(labels_index, f)

    # train_history.history is not JSON-serializable because it contains numpy arrays
    serializable_hist = {k: [float(e) for e in v] for k, v in train_history.history.items()}
    with open(os.path.join(artefacts_path, "train_output.json"), "w") as f:
        json.dump(serializable_hist, f)

    return scores[1], artefacts_path


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="Path to training dataset")
    parser.add_argument("config_path", help="Path to Yaml file specifying training parameters")
    parser.add_argument("artefacts_path", help="Folder where training artefacts will be persisted")
    parser.add_argument("add_timestamp", action='store_true',
                        help="Create artefacts in a sub folder with name equal to execution timestamp")

    args = parser.parse_args()

    with open(args.config_path, 'r') as config_f:
        train_params = yaml.safe_load(config_f.read())

    logger.info(f"Training model with parameters: {train_params}")

    train(args.dataset_path, train_params, args.artefacts_path, args.add_timestamp)
