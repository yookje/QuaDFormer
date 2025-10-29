import torch
from torch.utils.data import DataLoader, Dataset
from model import subsequent_mask

from tqdm import tqdm
from pprint import pprint

from encoder_lut import get_encoder_embedding
import time

"""
   src
   tgt
   src_mask
   tgt_mask
   tgt_y
   ntokens
   
   blank => -1
"""


class TSPDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        fuzzy_depth = 6,
    ):
        super(TSPDataset, self).__init__()
        self.data_path = data_path
        self.tsp_instances = []
        self.tsp_tours = []
        self.reversed_tsp_tours = []
        self.fuzzy_tsp_instances = []
        self.fuzzy_depth = fuzzy_depth

        start_time = time.time()
        self._readDataFile()
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time


        # self.tsp_instances = self.tsp_instances[:64] # delete
        # self.tsp_tours = self.tsp_tours[:64] # delete
        
        self.raw_data_size = len(self.tsp_instances)
        self.max_node_size = len(self.tsp_tours[0])
        
        self.src_fuzzy = []
        self.src = []
        self.tgt = []
        self.visited_mask = []
        self.tgt_y = []
        self.ntokens = []
        self._process()
        self.data_size = len(self.src)

        

        print()
        print("#### processing dataset... ####")
        print("data_path:", data_path)
        print("raw_data_size:", self.raw_data_size)
        print("max_node_size:", self.max_node_size)
        print("data_size:", self.data_size)
        print("processing time: ", elapsed_time)
        print()

    def _point2fuzzy(self, nodes_coord):
       
        fuzzy_encoding = get_encoder_embedding(nodes_coord, nodes_coord.size()[0], depth=self.fuzzy_depth)
        return fuzzy_encoding

    def _readDataFile(self):
        """
        read validation dataset from "https://github.com/Spider-scnu/TSP"
        """
        with open(self.data_path, "r") as fp:
            tsp_set = fp.readlines()
            for tsp in tsp_set: #[:100]
                tsp = tsp.split("output")
                tsp_instance = tsp[0].split()

                tsp_instance = [float(i) for i in tsp_instance]
                loc_x = torch.FloatTensor(tsp_instance[::2])
                loc_y = torch.FloatTensor(tsp_instance[1::2])
                tsp_instance = torch.stack([loc_x, loc_y], dim=1)
                self.tsp_instances.append(tsp_instance)

                tsp_tour = tsp[1].split()
                tsp_tour = [(int(i) - 1) for i in tsp_tour]
                reversed_tsp_tour = tsp_tour[1:].copy()
                reversed_tsp_tour.reverse()
                
                
                tsp_tour = torch.LongTensor(tsp_tour[:-1])
                self.tsp_tours.append(tsp_tour)

                fuzzy_tsp_instance = self._point2fuzzy(tsp_instance)
                self.fuzzy_tsp_instances.append(fuzzy_tsp_instance)

                
                reversed_tsp_tour = torch.LongTensor(reversed_tsp_tour)
                self.reversed_tsp_tours.append(reversed_tsp_tour)
        return

    def augment_xy_data_by_4_fold(self, xy_data, training):
        # xy_data.shape = [ N, 2]
        # x,y shape = [N, 1]

        
        x = xy_data[:, 0]
        y = xy_data[:, 1]

    

        dat1 = torch.stack([x, y], dim=1)
        dat2 = torch.stack([1 - x, y], dim=1)
        dat3 = torch.stack([x, 1 - y], dim=1)
        dat4 = torch.stack([1 - x, 1 - y], dim=1)

    

        # data_augmented.shape = [ N, 8]
        if training:
            data_augmented = torch.cat(
                (dat1, dat2, dat3, dat4), dim=1
            )
            
            return data_augmented

        # data_augmented.shape = [4*B, N, 2]
        data_augmented = torch.cat((dat1, dat2, dat3, dat4), dim=0)
        #print("data augmented2 ",data_augmented.size())
        return data_augmented
    
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

        
        return batch
    
    def data_augment3(self, graph_coord , processed_fuzzy, training = False):
        batch = torch.cat([graph_coord , processed_fuzzy], dim=-1)
        return batch
    
    def data_augment2(self, graph_coord , processed_fuzzy, training = False):
        batch = self.augment_xy_data_by_4_fold(graph_coord, training)
        theta = []
        for i in range(1):
            theta.append(
                torch.atan(batch[:,i * 2 + 1] / batch[:, i * 2]).unsqueeze(-1)
            )

        #batch = [N, 2(x,y coord) + 4(arc tan values of 4 folds)]
        #batch = torch.cat([graph_coord , theta[0]], dim=-1)
        batch = torch.cat([graph_coord , theta[0], processed_fuzzy], dim=-1)
        return batch

    def data_augment(self, graph_coord , processed_fuzzy, training = False):
        batch = self.augment_xy_data_by_4_fold(graph_coord, training)
        theta = []
        for i in range(4):
            theta.append(
                torch.atan(batch[:,i * 2 + 1] / batch[:, i * 2]).unsqueeze(-1)
            )

        #batch = [N, 2(x,y coord) + 4(arc tan values of 4 folds)]

        # just add arctan of 4 fold vertices
        batch = torch.cat([graph_coord , theta[0], theta[1], theta[2], theta[3], processed_fuzzy], dim=-1)

        # add 4 fold vertices
        #batch = torch.cat([batch , theta[0], theta[1], theta[2], theta[3], processed_fuzzy], dim=-1)
        return batch
    
    def _process(self):
        for fuzzy_instance, tsp_instance,  tsp_tour in tqdm(zip(self.fuzzy_tsp_instances, self.tsp_instances, self.tsp_tours)):
        #for fuzzy_instance, tsp_instance,  tsp_tour in tqdm(zip(self.fuzzy_tsp_instances, self.tsp_instances, self.tsp_tours)):
            self.ntokens.append(torch.LongTensor([1]))

            #modified
            enclut_instance = self.data_augment4(tsp_instance, fuzzy_instance, True)
            self.src_fuzzy.append(enclut_instance)

            #self.src_fuzzy.append(fuzzy_instance)
            #modified end

            self.src.append(tsp_instance)
            self.tgt.append(torch.tensor([tsp_tour[0]], dtype = torch.long))
            visited_mask = torch.zeros(self.max_node_size, dtype = torch.bool)
            visited_mask[self.tgt[-1][0]] = True
            self.visited_mask.append(visited_mask)
        return

    def __len__(self):
        return len(self.tsp_instances)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.visited_mask[idx], self.ntokens[idx], self.tsp_tours[idx], self.reversed_tsp_tours[idx], self.src_fuzzy[idx]


def make_tgt_mask(tgt):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != -1).unsqueeze(-2) # -1 equals blank
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def collate_fn(batch):
    src = [ele[0] for ele in batch]
    tgt = [ele[1] for ele in batch]
    visited_mask = [ele[2] for ele in batch]
    ntokens = [ele[3] for ele in batch]
    tsp_tours = [ele[4] for ele in batch]
    reversed_tsp_tours = [ele[5] for ele in batch]
    src_fuzzy = [ele[6] for ele in batch]

    tgt = torch.stack(tgt, dim=0)
    
    return {
        "src": torch.stack(src, dim=0),
        "tgt": tgt,
        "visited_mask": torch.stack(visited_mask, dim=0),
        "ntokens": torch.stack(ntokens, dim=0),
        "tgt_mask": make_tgt_mask(tgt),
        "tsp_tours": torch.stack(tsp_tours, dim=0),
        "reversed_tsp_tours": torch.stack(reversed_tsp_tours, dim=0),
        "src_fuzzy" : torch.stack(src_fuzzy, dim=0),
    }


if __name__ == "__main__":
    train_dataset = TSPDataset("./tsp20_test_concorde.txt")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for tsp_instances in tqdm(train_dataloader):
        print("tsp_instances")
        for k, v in tsp_instances.items():
            print(k, v)
        print()
        break
