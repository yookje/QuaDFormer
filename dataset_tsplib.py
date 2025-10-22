import torch
from torch.utils.data import DataLoader, Dataset
from model import subsequent_mask

from tqdm import tqdm
from pprint import pprint

import os
import tsplib95
from encoder_lut import get_encoder_embedding

class TSPDataset(Dataset):
    def __init__(
        self,
        data_path="../CycleFormer/tsplib/pr107.tsp",
    ):
        super(TSPDataset, self).__init__()
        self.data_path = data_path
        self.tsp_instances = []
        self.opt_tours = []
        self.norm_consts = []

        #modified for fuzzy
        self.src_fuzzy = []
        self.fuzzy_tsp_instances = []

        self._readDataFile()
        
        self.raw_data_size = len(self.tsp_instances)
        self.max_node_size = len(self.tsp_instances[0])
        
        self.src = []
        self.ntokens = []
        self._process()
        self.data_size = len(self.src)

        print()
        print("#### processing dataset... ####")
        print("data_path:", data_path)
        print("raw_data_size:", self.raw_data_size)
        print("max_node_size:", self.max_node_size)
        print("data_size:", self.data_size)
        print()

    def _point2fuzzy(self, nodes_coord, nodesize):
        fuzzy_encoding = get_encoder_embedding(nodes_coord, node_size=nodesize , depth=7)
        
        return fuzzy_encoding
    
    def _readDataFile(self):
        problem = tsplib95.load(self.data_path)
        opt_tour = tsplib95.load(self.data_path.split(".tsp")[0] + ".opt.tour")
        
        xs = []
        ys = []
        
        for idx, (x, y) in problem.node_coords.items():
            xs.append(x)
            ys.append(y)
            
        loc_x = torch.FloatTensor(xs)
        loc_y = torch.FloatTensor(ys)
        
        norm_const = torch.max(torch.cat([loc_x, loc_y]))
        tsp_instance = torch.stack([loc_x / norm_const, loc_y / norm_const], dim=1)
        self.tsp_instances.append(tsp_instance)
        self.norm_consts.append(norm_const)
        opt_tour = torch.tensor(opt_tour.tours)[0] - 1
        self.opt_tours.append(opt_tour)

        node_size = tsp_instance.size()[0]

        #print(f"\n tsp_instance {tsp_instance.size()}")
        

        fuzzy_tsp_instance = self._point2fuzzy(tsp_instance, nodesize=node_size)
        self.fuzzy_tsp_instances.append(fuzzy_tsp_instance)
      
        
        return
    def augment_xy_data_by_8_fold(self, xy_data, training):
        # xy_data.shape = [ N, 2]
        x = xy_data[:, 0]
        y = xy_data[:, 1]

        # 미리 계산된 값들
        one_minus_x = 1 - x
        one_minus_y = 1 - y

        # 변형된 데이터 미리 배열
        dats = [
            torch.stack([x, y], dim=1),
            torch.stack([one_minus_x, y], dim=1),
            torch.stack([x, one_minus_y], dim=1),
            torch.stack([one_minus_x, one_minus_y], dim=1),
            torch.stack([y, x], dim=1),
            torch.stack([one_minus_y, x], dim=1),
            torch.stack([y, one_minus_x], dim=1),
            torch.stack([one_minus_y, one_minus_x], dim=1)
        ]

        if training:
            # 학습 중일 때는 모든 변형을 결합
            data_augmented = torch.cat(dats, dim=1)
        else:
            # 학습 중이 아닐 때는 첫 4개의 변형만 결합
            data_augmented = torch.cat(dats[:4], dim=0)

        return data_augmented

    def data_augment4(self, graph_coord, processed_fuzzy, training=False):
        # 8배로 증강된 좌표 생성
        batch = self.augment_xy_data_by_8_fold(graph_coord, training)
        
        # theta 계산 (각도)
        x = batch[:, ::2]  # 짝수 번째 컬럼 (x 좌표)
        y = batch[:, 1::2]  # 홀수 번째 컬럼 (y 좌표)
        theta = torch.atan2(y, x)  # atan2는 더 안정적임 (x가 0일 때 처리됨)
        
        # theta와 다른 데이터 결합
        batch = torch.cat([batch, theta, processed_fuzzy], dim=-1)

        #batch = torch.cat([batch, theta], dim=-1)

        
        return batch
    
    def _process(self):
        for tsp_instance, fuzzy_instance in tqdm(zip( self.tsp_instances, self.fuzzy_tsp_instances)):
            ntoken = 1
            self.ntokens.append(torch.LongTensor([ntoken]))
            self.src.append(tsp_instance)

            #modified
            #print("\n tsp, guzzy", tsp_instance.size(), fuzzy_instance.size())
            enclut_instance = self.data_augment4(tsp_instance, fuzzy_instance, True)
            #enclut_instance=fuzzy_instance
            self.src_fuzzy.append(enclut_instance)
        return

    def __len__(self):
        return len(self.tsp_instances)

    def __getitem__(self, idx):
        return self.src[idx], self.ntokens[idx], self.norm_consts[idx], self.opt_tours[idx], self.src_fuzzy[idx]


def make_tgt_mask(tgt):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != -1).unsqueeze(-2) # -1 equals blank
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def collate_fn(batch):
    src = [ele[0] for ele in batch]
    ntokens = [ele[1] for ele in batch]
    norm_consts = [ele[2] for ele in batch]
    opt_tours = [ele[3] for ele in batch]
    src_fuzzy = [ele[4] for ele in batch]
    
    return {
        "src": torch.stack(src, dim=0),
        "ntokens": torch.stack(ntokens, dim=0),
        "norm_consts": torch.stack(norm_consts, dim=0),
        "opt_tours": torch.stack(opt_tours, dim=0),
        "src_fuzzy" : torch.stack(src_fuzzy, dim=0),
    }


if __name__ == "__main__":
    train_dataset = TSPDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for tsp_instances in tqdm(train_dataloader):
        for k, v in tsp_instances.items():
            print(k, v)
            

        print()
        break
