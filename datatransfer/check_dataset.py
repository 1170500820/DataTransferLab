"""
检查数据过程的正确性
"""
import sys
sys.path.append('..')

from rich.progress import track

from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from datatransfer.Datasets.RE import DuIE_CASREL_Dataset, casrel_dev_collate_fn, casrel_collate_fn
from datatransfer.temp_utils import dump_jsonl
from datatransfer.Models.RE.RE_settings import duie_relations_idx

def check_duie_dataset():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    result = []
    for data_type in ['train', 'valid']:
        dataset = DuIE_CASREL_Dataset(data_type=data_type, tokenizer=tokenizer)
        if data_type == 'train':
            batch_size = 1
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=casrel_collate_fn)
            for i, e in track(enumerate(dataloader), description=f'正在检查{data_type}-dataloader', total=len(dataloader)):
                check_result = {}
                inp, tgt = e
                origin = dataloader.dataset[batch_size * i: batch_size * i + batch_size]

                # subject gt
                start_gt, end_gt = (inp['subject_gt_start']==1).nonzero().tolist(), (inp['subject_gt_end']==1).nonzero().tolist()
                coords = []
                for _ in range(batch_size): coords.append(set())
                for gt in zip(start_gt, end_gt):
                    coords[gt[0][0]].add((gt[0][1], gt[1][1]))
                origin_coords = []
                for _ in range(batch_size): origin_coords.append(set())
                for i_origin, s in enumerate(origin):
                    for e_sub in s['all_subjects']:
                        origin_coords[i_origin].add(tuple(e_sub['subject_span']))
                flag = False
                for c_gt, c_origin in zip(coords, origin_coords):
                    if c_gt != c_origin:
                        flag = True
                if flag and False:
                    problems = {}
                    if len(start_gt) != len(end_gt):
                        problems['inequal'] = True
                    # problems['start_gt'] = start_gt
                    # problems['end_gt'] = end_gt
                    problems['label_coords'] = list(list(x) for x in coords)
                    problems['origin_coords'] = list(list(x) for x in origin_coords)
                    check_result['subject_gt'] = problems

                # subject label
                start_label, end_label = (tgt['subject_start_label'] == 1).nonzero().tolist(), \
                                         (tgt['subject_end_label'] == 1).nonzero().tolist()
                subject_problems = {}
                if len(start_label) != len(end_label) != batch_size:
                    subject_problems['absent or repeat label'] = (start_label, end_label, batch_size)
                else:
                    s_coords = []
                    for sl, el in zip(start_label, end_label):
                        assert sl[0] == el[0]
                        s_coords.append((sl[1], el[1]))
                    s_origin_coords = []
                    for i_origin, s in enumerate(origin):
                        s_origin_coords.append(tuple(s['subject_token_span']))
                    subject_flag = False
                    for es, eso in zip(s_coords, s_origin_coords):
                        if es != eso:
                            subject_flag = True
                            break
                    if subject_flag:
                        subject_problems['subject_dismatch'] = (s_coords, s_origin_coords)
                if subject_problems:
                    check_result['subject_problems'] = subject_problems



                # object label
                ostart_label, oend_label = (tgt['object_start_label'] == 1).nonzero().tolist(), \
                                           (tgt['object_end_label'] == 1).nonzero().tolist()
                object_problems = {}
                if len(ostart_label) == len(oend_label):
                    # 不相等的情况，在前面已经检查过了
                    ocoords = []
                    for _ in range(batch_size): ocoords.append(set())
                    for e_ostart, e_oend in zip(ostart_label, oend_label):
                        assert e_ostart[0] == e_oend[0]
                        ocoords[e_ostart[0]].add((e_ostart[1], e_ostart[2], e_oend[2]))
                    origin_ocoords = []
                    for _ in range(batch_size): origin_ocoords.append(set())
                    for i_origin, eo in enumerate(origin):
                        for e_ro in eo['relation_object']:
                            rel_idx = duie_relations_idx[e_ro['relation']]
                            origin_ocoords[i_origin].add((rel_idx, e_ro['object_token_span'][0], e_ro['object_token_span'][1]))

                    object_flag = False
                    for eo, eoo in zip(ocoords, origin_ocoords):
                        if eo != eoo:
                            object_flag = True
                            break
                    if object_flag:
                        object_problems['label_dismatch'] = (list(list(x) for x in ocoords), list(list(x) for x in origin_ocoords))
                if object_problems:
                    check_result['object_problems'] = object_problems


                if check_result:
                    check_result['data_type'] = 'train'
                    check_result['idx'] = i
                    result.append(check_result)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=casrel_dev_collate_fn)
    dump_jsonl(result, 'duie_dataloader_check_result.jsonl')


if __name__ == '__main__':
    check_duie_dataset()