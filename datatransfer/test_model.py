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

    def test_get_object_for_specific_index(self):
        indexes = [(2,4), (3,6)]
        encoded_text = torch.randn(2, 30, 768)
        object_start, object_end = self.model.get_object_for_specific_index(indexes, encoded_text)

        # 都为0～1的数值
        self.assertEqual(float(torch.sum(object_start< 0)), 0, '输出值应当为概率值')
        self.assertEqual(float(torch.sum(object_start > 1)), 0, '输出值应当为概率值')
        self.assertEqual(float(torch.sum(object_end < 0)), 0, '输出值应当为概率值')
        self.assertEqual(float(torch.sum(object_end > 1)), 0, '输出值应当为概率值')

    def test_get_object_for_indexes(self):
        indexes = [[(1, 3), (5, 6)], [(2, 3)], [(1, 2), (3, 4), (7, 8)], [(3,5), (9, 10)]]
        encoded_text = torch.randn(4, 30, 768)
        start, end = self.model.get_object_for_indexes(indexes, encoded_text)

        self.assertIsInstance(start, list, msg='输出的为list')
        self.assertIsInstance(end, list, msg='输出的为list')
        self.assertEqual(len(start), 4, msg='输出的list的长度应当和indexes的长度相同')
        self.assertEqual(len(end), 4, msg='输出的list的长度应当和indexes的长度相同')

        for e in start + end:
            self.assertEqual(float(torch.sum(e > 1)), 0, '输出的值应当为概率值')
            self.assertEqual(float(torch.sum(e < 0)), 0, '输出的值应当为概率值')
        for estart, eend, elength in zip(start, end, [2, 1, 3, 2]):
            self.assertEqual(estart.shape[0], elength)
            self.assertEqual(eend.shape[0], elength)


    def test_find_subject_spans(self):
        start = torch.randn(4, 30)
        end = torch.randn(4, 30)
        spans = self.model.find_subject_spans(start, end)

        self.assertIsInstance(spans, list)
        if len(spans) != 0:
            self.assertIsInstance(spans[0], list)
            if len(spans[0]) != 0:
                self.assertIsInstance(spans[0][0], tuple)

    def test_find_object_spans(self):
        start, end = torch.randn(4, 30, 48), torch.randn(4, 30, 48)
        ro = self.model.find_object_spans(start, end)

        self.assertIsInstance(ro, list)
        if len(ro) != 0:
            self.assertIsInstance(ro[0], list)
            if len(ro[0]) != 0:
                self.assertIsInstance(ro[0][0], tuple)


if __name__ == '__main__':
    unittest.main()