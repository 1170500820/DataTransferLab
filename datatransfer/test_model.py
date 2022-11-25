"""
对模型进行一些简单的单元测试
"""
import sys
sys.path.append('..')

import unittest
import torch

from datatransfer.information_extraction_model import CASREL2
from datatransfer.Models.RE.RE_settings import duie_relations


class Test_CASREL(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CASREL2({
            'class_cnt': len(duie_relations),
            'model_name': 'bert-base-chinese',
            'learning_rate': 1e-5,
            'linear_lr': 1e-5,
            'weight_decay': 0,
            'adam_epsilon': 1,
            'overfit': False
        })

    def test_get_subjects(self):
        pseudo_encoded_text = torch.randn(2, 10, 768)
        start_prob, end_prob = self.model.get_subjects(pseudo_encoded_text)

        # 都为0～1的数值
        self.assertEqual(float(torch.sum(start_prob < 0)), 0, '输出值应当为概率值')
        self.assertEqual(float(torch.sum(start_prob > 1)), 0, '输出值应当为概率值')
        self.assertEqual(float(torch.sum(end_prob < 0)), 0, '输出值应当为概率值')
        self.assertEqual(float(torch.sum(end_prob > 1)), 0, '输出值应当为概率值')

    # def test_get_objects_for_specific_subject(self):
    #     pas

    def test_get_object_for_specific_index(self):
        pass

    def test_get_object_for_indexes(self):
        pass

    def test_find_subject_spans(self):
        pass

    def test_find_object_spans(self):
        pass


if __name__ == '__main__':
    unittest.main()