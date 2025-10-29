import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from fuzzy_feature2 import get_fuzzy_features3_optimized
#from sklearn.metrics.pairwise import cosine_similarity

fuzzy8_degree_bins = torch.linspace(22.5,382.5,9)
fuzzy8_degree_bins[-1] = 360
appendix_bin=torch.tensor([0, 22.5])
fuzzy_degree_bins=torch.concat([fuzzy8_degree_bins, appendix_bin], dim=-1)


degree_bins = torch.linspace(0,360,13)




def get_count_matrix(dist_matrix, dist_max, dist_min=None):
    if dist_min!=None:
      
      heatmap1 = dist_matrix >= dist_min
      heatmap2 = dist_matrix < dist_max
      heatmap = heatmap1 * heatmap2
    
    else:
      heatmap = dist_matrix <=dist_max
      
    count_matrix = heatmap.sum(dim = -1)
    
    return count_matrix

def get_one_hot_encoding(count_nodes, num_categories):
    # 0 ~ num_categories-1 의 category에 할당됨.
    bins = torch.linspace(-1, count_nodes.max() + 1, num_categories + 1)
    category_indices = torch.bucketize(count_nodes, bins, right=False) - 1
    one_hot_categories = torch.nn.functional.one_hot(category_indices, num_categories)
    #print(one_hot_categories)
    return one_hot_categories

def get_embedding2(dist_matrix, node_size, bucket_size, rad1, rad2):
    
   
    range1 = rad1/node_size
    range2 = rad2/node_size


    count_matrix1 = get_count_matrix(dist_matrix,  range1) # range1 구간의 개수 counting, [N]
    count_matrix2 = get_count_matrix(dist_matrix,  range2) # range2 구간의 개수 counting, [N]

    #print("count_matrix1 ", count_matrix1)
    #print("count_matrix2 ", count_matrix2)
    
    one_hot1 = get_one_hot_encoding(count_matrix1-1,  bucket_size)
    one_hot2 = get_one_hot_encoding(count_matrix2-1,  bucket_size)
    
    embedding2 = torch.concat([one_hot1, one_hot2], dim = -1)
    return embedding2


def get_embedding4(points, dist_matrix, node_size, bucket_size, rad2):

    x_i, y_i = points[:, 0].view(node_size, 1), points[:, 1].view(node_size, 1)
    x_j, y_j = points[:, 0].view(1, node_size), points[:, 1].view(1, node_size)
    
    range3 = rad2/node_size
    
    # 거리 조건 필터링
    mask = dist_matrix <= range3
    valid_mask = torch.empty(mask.size())
    valid_mask[mask == True] = 1
    valid_mask[mask==False] = -math.inf

    
    # 각도 계산 (-180 ~ 180 범위) 및 0~360 변환
    angles = torch.atan2(y_j - y_i, x_j - x_i) * (180 / torch.pi)
    angles[angles < 0] += 360

    angles = angles * valid_mask

    # 자기자신 제외
    angles.fill_diagonal_(-math.inf)

    #print("after angles", angles)


    count_matrix2 = get_count_matrix(angles,  degree_bins[1], degree_bins[0])
    count_matrix3 = get_count_matrix(angles,  degree_bins[2], degree_bins[1])
    count_matrix4 = get_count_matrix(angles,  degree_bins[3], degree_bins[2])
    count_matrix5 = get_count_matrix(angles,  degree_bins[4], degree_bins[3])
    count_matrix6 = get_count_matrix(angles,  degree_bins[5], degree_bins[4])
    count_matrix7 = get_count_matrix(angles,  degree_bins[6], degree_bins[5])
    count_matrix8 = get_count_matrix(angles,  degree_bins[7], degree_bins[6])
    count_matrix9 = get_count_matrix(angles,  degree_bins[8], degree_bins[7])
    count_matrix10 = get_count_matrix(angles,  degree_bins[9], degree_bins[8])
    count_matrix11 = get_count_matrix(angles,  degree_bins[10], degree_bins[9])
    count_matrix12 = get_count_matrix(angles,  degree_bins[11], degree_bins[10])
    count_matrix13 = get_count_matrix(angles,  degree_bins[12], degree_bins[11])

    one_hot2 = get_one_hot_encoding(count_matrix2,  bucket_size)
    one_hot3 = get_one_hot_encoding(count_matrix3,  bucket_size)
    one_hot4 = get_one_hot_encoding(count_matrix4,  bucket_size)
    one_hot5 = get_one_hot_encoding(count_matrix5,  bucket_size)
    one_hot6 = get_one_hot_encoding(count_matrix6,  bucket_size)
    one_hot7 = get_one_hot_encoding(count_matrix7,  bucket_size)

    one_hot8 = get_one_hot_encoding(count_matrix8,  bucket_size)
    one_hot9 = get_one_hot_encoding(count_matrix9,  bucket_size)
    one_hot10 = get_one_hot_encoding(count_matrix10,  bucket_size)
    one_hot11 = get_one_hot_encoding(count_matrix11,  bucket_size)
    one_hot12 = get_one_hot_encoding(count_matrix12,  bucket_size)
    one_hot13 = get_one_hot_encoding(count_matrix13,  bucket_size)

    embedding3 = torch.concat([one_hot2, one_hot3, one_hot4, one_hot5, one_hot6,\
     one_hot7, one_hot8, one_hot9 , one_hot10, one_hot11, one_hot12, one_hot13], dim = -1)#], dim = -1)#

    return embedding3




def get_encoder_embedding(tsp_instance, node_size, depth=6,  bucket_size=4):
    
    dist_matrix = torch.cdist(tsp_instance, tsp_instance)

    embedding1 = get_fuzzy_features3_optimized(tsp_instance, depth)

      
    #op1
    if node_size <= 75 :
       rad1 = 7
       rad2 = 19
    elif (node_size > 75) & (node_size <= 250):
       rad1 = 10
       rad2 = 25
    elif (node_size > 250) & (node_size <= 750):
       rad1 = 22
       rad2 = 52
    elif (node_size > 750) & (node_size <= 2000):
       rad1 = 31
       rad2 = 73
    #"""
    


    embedding2 = get_embedding2(dist_matrix, node_size, bucket_size, rad1, rad2)
    embedding3 = get_embedding4(tsp_instance, dist_matrix, node_size, bucket_size, rad2)

    embedding = torch.concat([embedding1, embedding2,  embedding3], dim = -1)
    return embedding
    

    #return embedding1

  
    
    

