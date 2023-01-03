import unittest
import pandas as pd
from unittest.mock import MagicMock

'''
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        baseTextC.ft qui donne return value = MagicMock(return_value={exemple type retournée par ft})
        self.assertEqual(baseTextC.ft à tester avec le nom de la ft en haut(),{result attendu})
'''

from preprocessing.preprocessing import utils
# mock : simuler return
# MagicMock tester uniquement la fonction et pas les fonctions auxquelles on fait appeller,
# python n'évalue que la fonction actuelle
class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        baseTextC._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(baseTextC._get_num_train_samples(), 80)


    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        baseTextC._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_batches will return  _get_num_train_samples / batche = 4
        self.assertEqual(baseTextC._get_num_train_batches(), 4)


    def test__get_num_test_batches(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        baseTextC._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_test_batches will return  _get_num_test_samples / batche = 1
        self.assertEqual(baseTextC._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        # retourne le dico
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        baseTextC._get_label_list = MagicMock(return_value=["a","b","c"]) #test sur liste
        # we assert that get_index_to_label_map will return labels_list[i]
        self.assertEqual(baseTextC.get_index_to_label_map(),{0:"a",1:"b",2:"c"}) # on veut tester get_index_to_label_map
    # on veut tester la fonction get_index_to_label_map qui retourne l'index
    # pour ca je dois créer une liste et vérifier que get_index_to_label_map retourne le dictionnaire associé
    ## convert list to dictionary with indexes as keys

    def test_index_to_label_and_label_to_index_are_identity(self):
        # inverse les clées
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        baseTextC.get_index_to_label_map = MagicMock(return_value={0:"a",1:"b",2:"c"})
        self.assertEqual(baseTextC.get_label_to_index_map(),{"a":0,"b":1,"c":2})

    def test_to_indexes(self):
        #prend list label et retourne liste index
        # creer des labels et ca doit retourner la liste d'index
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        baseTextC._get_label_list = MagicMock(return_value=["a","b","c"])
        self.assertEqual(baseTextC.to_indexes(["a","b","c"]),[0,1,2])


class TestLocalTextCategorizationDataset(unittest.TestCase):

    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path",1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)


    def test__get_num_samples_is_correct(self):
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        baseTextC._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(baseTextC._get_num_samples(),100)


# renvoi x et y
# on fait insertioon sur shape de x, de shape :
    #assert de size pour verifier shape x
    #y : 2,none ?

# verifier que batch size est bonne size
    def test_get_train_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2','id_3', 'id_1', 'id_2','id_3'],
            'tag_name': ['tag_a', 'tag_a','tag_a', 'tag_a', 'tag_a','tag_a'],
            'tag_id': [1, 2,3, 1, 2,3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2','title_3', 'title_1', 'title_2','title_3']
        }))
        data = utils.LocalTextCategorizationDataset("fake_path",batch_size=2,train_ratio=0.5, min_samples_per_label=1)
        x,y = data.get_train_batch()
        self.assertEqual(x.shape,(2,))
# si ca raise error c'est car pas assez test batch size


    def test_get_test_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2','id_3', 'id_1', 'id_2','id_3'],
            'tag_name': ['tag_a', 'tag_a','tag_a', 'tag_a', 'tag_a','tag_a'],
            'tag_id': [1, 2,3, 1, 2,3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2','title_3', 'title_1', 'title_2','title_3']
        }))
        # en local
        data = utils.LocalTextCategorizationDataset("fake_path",batch_size=1,train_ratio=0.8, min_samples_per_label=1)
        x,y = data.get_test_batch()
        self.assertEqual(x.shape,(1,)) #tuple


'''
        baseTextC = utils.BaseTextCategorizationDataset(20, 0.8)
        baseTextC.ft qui donne return value = MagicMock(return_value={exemple type retournée par ft})
        self.assertEqual(baseTextC.ft à tester avec le nom de la ft en haut(),{result attendu})
'''