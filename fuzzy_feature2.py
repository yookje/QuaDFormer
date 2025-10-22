import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# ============ 원본 코드 (변경 없음) ============
mid_points_matrix=[[[-1/2**(d+2),-1/2**(d+2)],[1/2**(d+2),-1/2**(d+2)],[-1/2**(d+2),1/2**(d+2)],[1/2**(d+2),1/2**(d+2)]]for d in range(10)]
band_debth_matrix=[1/30, 1/30, 1/30, 1/30, 1/30, 1/30, 1/30, 1/30, 1/30,1/30]
grid_size_matrix = [1,1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256,1/512]
box_to_binary = {
    2: [0, 0, 0, 1],
    3: [0, 1, 0, 0],
    1: [0, 0, 1, 0],
    4: [1, 0, 0, 0]
}
half_band_matrix= [band_debth_matrix[ii]*grid_size_matrix[ii] for ii in range(len(grid_size_matrix))]

def get_band_ratio(depth):
    return band_debth_matrix[depth-1]

# ============ 원본 함수들 ============
def process_point_original(point_idx, coordinates, mid_points):
    point = coordinates[point_idx]
    bit_result = []
    
    for depth in range(len(mid_points[point_idx])):
        mid_point = mid_points[point_idx][depth]
        half_band_width = half_band_matrix[depth]
        bits = get_four_bits_original(point, mid_point, half_band_width)
        bit_result.extend(bits)
    return bit_result

def get_four_bits_original(point, mid_point, half_band_width):
    mid_x, mid_y = mid_point
    up = point[1] > (mid_y - half_band_width)
    down = point[1] < (mid_y + half_band_width)
    right = point[0] > (mid_x - half_band_width)
    left = point[0] < (mid_x + half_band_width)
    
    temp = [left & right & up, right & down & up, left & right & down, left & up & down]
    return [int(element) for element in temp]

def apply_shifts_for_column(initial_center, shifts_column):
    center = initial_center
    centers_for_column = []
    for shift in shifts_column:
        center = apply_single_shift(center, shift)
        centers_for_column.append(center)
    return centers_for_column

def apply_single_shift(center, shift):
    return (center[0] + shift[0], center[1] + shift[1])

def get_fuzzy_features3_original(coordinates, depth):
    epsilon = 1e-7
    coordinates = torch.clamp(coordinates, min=0.0, max=1.0 - epsilon)
    coordinates = np.array(coordinates)
    
    new_coordinates = coordinates * (2**(depth+1))
    xs = new_coordinates[:,0]
    ys = new_coordinates[:,1]
    
    rights = np.array([(xs%(2**(value+1))//(2**value)).astype(int) for value in range(depth,0,-1)])
    ups = np.array([(ys%(2**(value+1))//(2**value)).astype(int) for value in range(depth,0,-1)])
    
    boxes = 1 + rights + 2*ups
    boxes_transposed = boxes.T
    
    binary_representation = np.array([[box_to_binary[box] for box in row] for row in boxes_transposed])
    extended_binary_representation = np.array([np.concatenate(row) for row in binary_representation])
    
    grid_coord_shifts = [[mid_points_matrix[d][b-1] for b in row] for d,row in enumerate(boxes)]
    
    initial_center = [0.5, 0.5]
    final_results = []
    
    for column_index in range(len(grid_coord_shifts[0])):
        shifts_column = [row[column_index] for row in grid_coord_shifts if column_index < len(row)]
        column_results = apply_shifts_for_column(initial_center, shifts_column)
        bef = [(0.5, 0.5)]
        bef.extend(column_results)
        bef = bef[:-1]
        final_results.append(bef)
    
    n = len(coordinates)
    results = [None] * n
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_point_original, i, coordinates, final_results) for i in range(n)]
        for future in futures:
            idx = futures.index(future)
            results[idx] = future.result()
    
    fin_result = []
    for row_a, row_b in zip(extended_binary_representation, results):
        new_row = []
        for i in range(0, len(row_a), 4):
            new_row.extend(row_a[i:i+4])
            new_row.extend(row_b[i:i+4])
        fin_result.append(new_row)
    
    return torch.tensor(fin_result)

# ============ 전역 캐싱 (사전 계산) ============
# 최대 depth 10까지 미리 계산해둠
MAX_DEPTH = 10
POWERS_CACHE = {d: np.array([2**i for i in range(d, 0, -1)]) for d in range(1, MAX_DEPTH + 1)}
MODS_CACHE = {d: POWERS_CACHE[d] * 2 for d in range(1, MAX_DEPTH + 1)}
SCALES_CACHE = {d: 2**(d+1) for d in range(1, MAX_DEPTH + 1)}

# box_to_binary를 NumPy 배열로 변환 (인덱싱 속도 향상)
BINARY_LOOKUP = np.array([[0, 0, 1, 0],  # box 1
                          [0, 0, 0, 1],  # box 2
                          [0, 1, 0, 0],  # box 3
                          [1, 0, 0, 0]]) # box 4

# ============ 최적화된 버전 ============

def get_fuzzy_features3_optimized(coordinates, depth):
    """최적화된 버전 - 전역 캐싱과 벡터화"""
    epsilon = 1e-7
    coordinates = torch.clamp(coordinates, min=0.0, max=1.0 - epsilon)
    coordinates_np = np.array(coordinates)
    n_points = len(coordinates_np)
    
    # 캐싱된 값 사용
    scale = SCALES_CACHE[depth]
    powers = POWERS_CACHE[depth]
    mods = MODS_CACHE[depth]
    
    # 좌표 변환
    new_coordinates = coordinates_np * scale
    xs = new_coordinates[:, 0]
    ys = new_coordinates[:, 1]
    
    # 벡터화된 연산 - 캐싱된 powers와 mods 사용
    rights = np.zeros((depth, n_points), dtype=int)
    ups = np.zeros((depth, n_points), dtype=int)
    
    for i in range(depth):
        rights[i] = (xs % mods[i]) // powers[i]
        ups[i] = (ys % mods[i]) // powers[i]
    
    boxes = 1 + rights + 2 * ups
    boxes_transposed = boxes.T
    
    # 캐싱된 BINARY_LOOKUP 사용
    binary_representation = BINARY_LOOKUP[boxes_transposed - 1]
    extended_binary_representation = binary_representation.reshape(n_points, -1)
    
    # 중점 계산 - 캐싱된 mid_points_matrix 활용
    all_midpoints = np.zeros((n_points, depth, 2))
    
    for col_idx in range(n_points):
        center = np.array([0.5, 0.5])
        all_midpoints[col_idx, 0] = center
        
        for d in range(1, depth):
            shift_idx = boxes[d-1, col_idx] - 1
            shift = mid_points_matrix[d-1][shift_idx]
            center = center + np.array(shift)
            all_midpoints[col_idx, d] = center
    
    # 벡터화된 bit 계산 - half_band_matrix 슬라이싱
    half_bands = np.array(half_band_matrix[:depth])
    all_bit_results = np.zeros((n_points, depth * 4), dtype=int)
    
    for point_idx in range(n_points):
        point_x = coordinates_np[point_idx, 0]
        point_y = coordinates_np[point_idx, 1]
        
        # 모든 depth에 대해 벡터화
        mid_x = all_midpoints[point_idx, :, 0]
        mid_y = all_midpoints[point_idx, :, 1]
        
        up = point_y > (mid_y - half_bands)
        down = point_y < (mid_y + half_bands)
        right = point_x > (mid_x - half_bands)
        left = point_x < (mid_x + half_bands)
        
        # 벡터화된 bit 할당
        for d in range(depth):
            base_idx = d * 4
            all_bit_results[point_idx, base_idx] = left[d] & right[d] & up[d]
            all_bit_results[point_idx, base_idx + 1] = right[d] & down[d] & up[d]
            all_bit_results[point_idx, base_idx + 2] = left[d] & right[d] & down[d]
            all_bit_results[point_idx, base_idx + 3] = left[d] & up[d] & down[d]
    
    # 최종 결과 조합 - NumPy 벡터화
    fin_result = np.zeros((n_points, depth * 8), dtype=int)
    
    for j in range(depth):
        base_idx = j * 8
        bin_start = j * 4
        bit_start = j * 4
        
        fin_result[:, base_idx:base_idx+4] = extended_binary_representation[:, bin_start:bin_start+4]
        fin_result[:, base_idx+4:base_idx+8] = all_bit_results[:, bit_start:bit_start+4]
    
    return torch.tensor(fin_result)

# ============ 검증 함수 ============
def verify_optimization():
    """최적화 전후 결과가 동일한지 검증"""
    # 테스트 데이터
    #coordinates = torch.tensor([[0.9062, 0.1269], [0.9501, 0.5458]])
    coordinates = torch.tensor([[0.9362, 0.1189], [0.9511, 0.5459]])
    depth = 6
    
    print("=" * 60)
    print("검증 시작: coordinates = torch.tensor([[0.9062, 0.1269], [0.9501, 0.5458]])")
    print("Depth = 6")
    print("=" * 60)
    
    # 원본 알고리즘 실행
    print("\n[원본 알고리즘 실행 중...]")
    start = time.time()
    result_original = get_fuzzy_features3_original(coordinates, depth)
    time_original = time.time() - start
    print(f"원본 실행 시간: {time_original:.6f}초")
    print(f"원본 결과 shape: {result_original.shape}")
    
    # 최적화 알고리즘 실행
    print("\n[최적화 알고리즘 실행 중...]")
    start = time.time()
    result_optimized = get_fuzzy_features3_optimized(coordinates, depth)
    time_optimized = time.time() - start
    print(f"최적화 실행 시간: {time_optimized:.6f}초")
    print(f"최적화 결과 shape: {result_optimized.shape}")
    
    # 결과 비교
    print("\n" + "=" * 60)
    print("결과 검증:")
    are_equal = torch.equal(result_original, result_optimized)
    print(f"결과 동일 여부: {are_equal}")
    
    if are_equal:
        print("✓ 최적화가 성공적으로 완료되었습니다!")
        if time_optimized > 0:
            print(f"속도 향상: {time_original/time_optimized:.2f}배")
    else:
        print("✗ 결과가 다릅니다.")
        diff = torch.abs(result_original.float() - result_optimized.float())
        print(f"최대 차이: {torch.max(diff)}")
        
    print("\n전체 출력 (원본):")
    print(result_original)
    print("\n전체 출력 (최적화):")
    print(result_optimized)
    
    return are_equal, time_original, time_optimized



if __name__ == "__main__":
    # 검증 실행
    success, t_orig, t_opt = verify_optimization()
    
   