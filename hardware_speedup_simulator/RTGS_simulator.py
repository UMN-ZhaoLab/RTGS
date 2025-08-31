import json
import numpy as np
import math
import matplotlib.pyplot as plt
resources_guassian= 16
Gaussian_all= 37819 #改
def simulation(max_pe_time,width, height,down):
    # 计算延迟
    # latency = max_pe_time / 1000000000  # 转换为秒
    # print(f"⏳ Total latency: {latency:.2f} seconds")
    frequency = 500
    gmu_dominate=0
    
    rendering_latency=((18+max_pe_time*(12+8)+16*(width*height/(down*down*256)))+4*gmu_dominate)*(1/(frequency*1000000))# Leshu O:30 #+GMU:20 #+BP: 8
    #rendering_latency=((18+max_pe_time*(12))*(1/(frequency*1000000))
    RTGS_latency=rendering_latency
    conv2D_3D_cycles=5
    conv3D_R_cycles=5
    R_q_cycles=5
    conv2D_T_cycles=6
    T_J_cycles=3
    J_3D_cycles=9
    color_SH_cycles=1
    position2D_3D_cycles=11
    SH_position_cycles=7
    
    #resources_guassian= 32
    cycles_BP_1 = (
    max(conv2D_3D_cycles, conv3D_R_cycles, R_q_cycles) * 
    (int(Gaussian_all / resources_guassian) + 1) + 
    conv2D_3D_cycles + conv3D_R_cycles + R_q_cycles - 
    max(conv2D_3D_cycles, conv3D_R_cycles, R_q_cycles))

    cycles_BP_2 = (
    max(conv2D_T_cycles, T_J_cycles, J_3D_cycles) * 
    (int(Gaussian_all / resources_guassian) + 1) + 
    conv2D_T_cycles + T_J_cycles + J_3D_cycles - 
    max(conv2D_T_cycles, T_J_cycles, J_3D_cycles))

    cycles_BP_3 = (
    max(color_SH_cycles, SH_position_cycles) * 
    (int(Gaussian_all / resources_guassian) + 1) + 
    color_SH_cycles + SH_position_cycles - 
    max(color_SH_cycles, SH_position_cycles))

    cycles_BP_4 = (
    position2D_3D_cycles * 
    (int(Gaussian_all / resources_guassian) + 1))

    preprocessing_latency = (
    1 * 2 + 5*(int(Gaussian_all / resources_guassian) + 1)+
    max(cycles_BP_1, cycles_BP_2, cycles_BP_3, cycles_BP_4))
    RTGS_latency += preprocessing_latency * (1/(frequency*1000000))
    return RTGS_latency

def area():
    add_sub_area=1360
    mul_area=1640
    exp_area=13600
    div_area=13600
    pow_area=13600
    #resources_guassian= 32
    resources_pixels=256
    adder_tree=(16+16+8*1.5)*add_sub_area*4/(1.6*1.6*1000000)+(36*1024/128*706.584)/1000000
    memory_area=(161*1024/128*706.584)/1000000
    WSU_area=(80*add_sub_area*16/(1.6*1.6*1000000))+(6*1024/128*706.584)/1000000
    RTGS_area=memory_area+adder_tree
    rendering_A_area=5*add_sub_area+9*mul_area+1*exp_area
    rendering_C_area=2*add_sub_area+2*mul_area
    rendering_get_loss_area=4*add_sub_area+4*pow_area+3*add_sub_area
    rendering_loss_2Dcolor_area=1*mul_area
    loss_pixelalpha_area=16*add_sub_area+12*mul_area
    pixelalpha_distribution_area=4*add_sub_area+7*mul_area
    distribution_2Dconv_position_area=11*mul_area
    rendering_area=(rendering_A_area+rendering_C_area+rendering_get_loss_area+rendering_loss_2Dcolor_area+loss_pixelalpha_area
                    +loss_pixelalpha_area+pixelalpha_distribution_area+distribution_2Dconv_position_area)*resources_pixels
    RTGS_area+=(rendering_area/(1000000.0*1.6*1.6))
    conv2D_3D_area=15*add_sub_area+45*mul_area
    conv3D_R_area=9*add_sub_area+9*mul_area
    R_q_area=20*add_sub_area+22*mul_area
    conv2D_T_area=18*add_sub_area+33*mul_area
    T_J_area=12*add_sub_area+8*mul_area
    J_3D_area=5*add_sub_area+20*mul_area+1*div_area
    color_SH_area=3*mul_area
    position2D_3D_area=16*add_sub_area+25*mul_area+1*div_area
    SH_position_area=20*add_sub_area+22*mul_area
    position_camera_pose_area=48*add_sub_area+54*mul_area
    preprocessing_area=(conv2D_3D_area+conv3D_R_area+R_q_area+conv2D_T_area+T_J_area+J_3D_area+color_SH_area+position2D_3D_area+SH_position_area\
+position_camera_pose_area)*resources_guassian
    RTGS_area += (preprocessing_area/(1000000.0*1.6*1.6)+WSU_area)
    print(f"adder tree count for: {adder_tree/(RTGS_area):.10f}")
    print(f"WSU area count for: {WSU_area/RTGS_area:.10f}")
    return RTGS_area
    
def energy(sum_all_gaussian,width, height,down):
    sum_Gaussian_all = sum_all_gaussian
    add_sub_energy = 0.25*2  # pJ
    mul_energy = 0.68*2
    exp_energy = 2.74*2
    div_energy=2.74*2

    rendering_A_energy = (5 * add_sub_energy + 9 * mul_energy + exp_energy) * sum_Gaussian_all
    rendering_C_energy= (2 * add_sub_energy + 2 * mul_energy) * sum_Gaussian_all
    rendering_get_loss_energy = (4 * add_sub_energy + 4 * exp_energy + 3 * add_sub_energy) * width * height/ (down * down)
    rendering_loss_2Dcolor_energy = (1 * mul_energy) * sum_Gaussian_all
    loss_pixelalpha_energy = (16 * add_sub_energy + 12 * mul_energy) * sum_Gaussian_all
    pixelalpha_distribution_energy = (4 * add_sub_energy + 7 * mul_energy) * sum_Gaussian_all
    distribution_2Dconv_position_energy = (11 * mul_energy) * sum_Gaussian_all
    adder_color_conv_position_energy = (add_sub_energy) * sum_Gaussian_all
    conv2D_3D_energy = (15 * add_sub_energy + 45 * mul_energy) * Gaussian_all
    conv3D_R_energy = (9 * add_sub_energy + 9 * mul_energy) * Gaussian_all
    R_q_energy = (20 * add_sub_energy + 22 * mul_energy) * Gaussian_all
    conv2D_T_energy = (18 * add_sub_energy + 33 * mul_energy) * Gaussian_all
    T_J_energy = (12 * add_sub_energy + 8 * mul_energy) * Gaussian_all
    J_3D_energy = (5 * add_sub_energy + 20 * mul_energy + div_energy) * Gaussian_all
    color_SH_energy = (3 * mul_energy) * Gaussian_all
    position2D_3D_energy = (16 * add_sub_energy + 25 * mul_energy + div_energy) * Gaussian_all
    SH_position_energy = (20 * add_sub_energy + 22 * mul_energy) * Gaussian_all
    position_camera_pose_energy = (48 * add_sub_energy + 54 * mul_energy) * Gaussian_all
    position_adder_energy=9*(15*add_sub_energy)*(Gaussian_all/resources_guassian)
    computing_energy = (rendering_A_energy + rendering_C_energy + rendering_get_loss_energy + rendering_loss_2Dcolor_energy + loss_pixelalpha_energy + pixelalpha_distribution_energy + distribution_2Dconv_position_energy + adder_color_conv_position_energy + conv2D_3D_energy + conv3D_R_energy + R_q_energy + conv2D_T_energy + T_J_energy + J_3D_energy + color_SH_energy + position2D_3D_energy + SH_position_energy + position_camera_pose_energy+position_adder_energy) 
    RTGS_energy = computing_energy
    
    write_SRAM_energy = 0.092867 * 16
    read_SRAM_energy = 0.0950 * 16
    GSC=sum_Gaussian_all*10*read_SRAM_energy*2+sum_Gaussian_all*10*write_SRAM_energy*2*8
    RTGS_energy += GSC
    pixel=width*height/(down*down)*2*(read_SRAM_energy+write_SRAM_energy)
    RTGS_energy += pixel
    buffer_2D= sum_Gaussian_all*10*read_SRAM_energy*2+sum_Gaussian_all*10*write_SRAM_energy*2*8
    RTGS_energy += buffer_2D
    buffer_3D= Gaussian_all*14*read_SRAM_energy+Gaussian_all*14*write_SRAM_energy*8
    RTGS_energy += buffer_3D
    Reuse_buffer= sum_Gaussian_all*1*read_SRAM_energy+sum_Gaussian_all*1*write_SRAM_energy*8
    RTGS_energy += Reuse_buffer
    Stage_buffer= (sum_Gaussian_all*8*read_SRAM_energy+sum_Gaussian_all*8*write_SRAM_energy)*3*8
    RTGS_energy += Stage_buffer
    
    return RTGS_energy

# === Step 0: Load early termination count file ===


width = 1752#改
height = 1160#改
iteration=1#改
tile_size_1 = 16#Leshu 改 8
tile_size_2 = 16#Leshu 改 8
downsample_stride = 4#Leshu 改 2
RTGS_latency=0
RTGS_area=0
RTGS_energy=0
RTGS_area=area()
print(f"📐 Total area: {RTGS_area:.10f} mm²")
#exit(1)
# tile 尺寸（大tile，用于下采样）
for i in range(iteration):
    if i//8==0:#改
        # Use the transformed data file instead of hardcoded path
        try:
            with open("transformed_data.json", "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            # Fallback to default data structure
            data = {"pixels": {}}
    else:
        try:
            with open("transformed_data.json", "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            # Fallback to default data structure
            data = {"pixels": {}}

    # 下采样步长（把16x16 → 4x4）

    group = 2  # 用于分组配对平均

    pixel_counts = data["pixels"]

    # === Step 1: Padding to tile size ===
    pad_width = (tile_size_1 - (width % tile_size_1)) % tile_size_1
    pad_height = (tile_size_2 - (height % tile_size_2)) % tile_size_2
    new_width = width + pad_width
    new_height = height + pad_height

    # === Step 2: Build padded count map ===
    count_map = np.zeros((new_height, new_width), dtype=np.int32)
    for key, value in pixel_counts.items():
        u, v = map(int, key.split('_'))
        if isinstance(value, list):
            count_map[v][u] = len(value)
        else:
            count_map[v][u] = 0  # 默认填 0，按需调整

    # === Step 3: Tile analysis and async scheduling simulation ===
    tiles_y = new_height // tile_size_2
    tiles_x = new_width // tile_size_1
    sum_all_gaussian = 0
    sum_raw_max = 0
    sum_group_max = 0
    sum_avg_max = 0
    tile_exec_times = []
    count_max=0
    with open("imbalance.txt", "w") as f:
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                y_start = ty * tile_size_2
                y_end = y_start + tile_size_2
                x_start = tx * tile_size_1
                x_end = x_start + tile_size_1

                tile = count_map[y_start:y_end, x_start:x_end]

                # === Downsample to 4x4 ===
                tile = tile[::downsample_stride, ::downsample_stride]  # e.g., 16x16 -> 4x4

                # === 原始最大值与平均值 ===
                raw_max = np.max(tile)
                avg_max = math.ceil(np.mean(tile))
                all_gaussian=np.sum(tile)
                if (raw_max>count_max):
                    count_max=raw_max
                
                sum_raw_max += raw_max
                sum_avg_max += avg_max

                # === 分组最大值（最大+最小配对平均）===
                vals = list(tile.flatten())
                vals.sort()
                group_avgs = []
                num_groups = len(vals) // group
                if group % 2 != 0:
                    raise ValueError("Group size must be even")

                for _ in range(num_groups):
                    if len(vals) < group:
                        break
                    vals.sort()
                    min_vals = vals[:group // 2]
                    max_vals = vals[-(group // 2):]
                    for v in min_vals + max_vals:
                        vals.remove(v)
                    pair_avgs = [(min_vals[i] + max_vals[i]) / 2 for i in range(group // 2)]
                    group_avg = math.ceil(np.mean(pair_avgs))
                    group_avgs.append(group_avg)

                group_max = math.ceil(max(group_avgs)) if group_avgs else 0
                sum_group_max += group_max
                sum_all_gaussian += all_gaussian
                tile_exec_times.append(group_max) #Leshu GMU: raw_max +WSU: group_max
                
                #f.write(f"Tile ({ty}, {tx}): ILF=({raw_max/avg_max})\n")

    # === Step 4: Async PE scheduling (16 processing units) ===
    num_pes = 16#改 #Leshu GMU: 1 +WSU:16
    pe_loads = [0 for _ in range(num_pes)]  # 每个处理单元的累计时间
    with open("assign.txt", "w") as f:
        for exec_time in tile_exec_times:
            min_index = pe_loads.index(min(pe_loads))  # 找当前最空的 PE
            pe_loads[min_index] += exec_time
            
            f.write(f"PE {min_index:02d} assigned {exec_time} cycles, total load: {pe_loads[min_index]} cycles\n")

    # === Step 5: Output Summary ===
    print(count_max)
    print("\n📊 Summary:")
    print(f"🧮 Sum of raw max values: {sum_raw_max}")
    print(f"🧮 Sum of max 2x1 group means: {sum_group_max:.2f}")
    print(f"🧮 Sum of all Gaussians : {sum_all_gaussian:.2f}")
    if sum_group_max > 0:
        print(f"🚀 Improvement Ratio (Raw Max / 2x1 Group Max): {sum_raw_max / sum_group_max:.2f}x")
        print(f"🚀 Improvement Ratio (Raw Max / Avg Max): {sum_raw_max / sum_avg_max:.2f}x")
    else:
        print("⚠️ Group max is zero, cannot compute improvement ratio.")

    print(f"\n🧩 Padding applied: width +{pad_width}, height +{pad_height}")

    # === Step 6: PE Load Summary ===
    # print("\n⚙️ PE Execution Time Summary:")
    # for i, load in enumerate(pe_loads):
    #     print(f"🔧 PE {i:02d}: {load} cycles")

    max_pe_time = max(pe_loads)
    print(f"\n⏱️ Total execution time (max PE load): {max_pe_time} cycles")
    RTGS_latency+=simulation(max_pe_time, width, height, downsample_stride)
    RTGS_energy+=energy(sum_all_gaussian,width, height, downsample_stride)
print(f"⏳ Total latency: {RTGS_latency:.10f} seconds")
print(f"💡 Total energy: {RTGS_energy/1000000000000:.10f} J")
print(f"Power: {RTGS_energy/(1000000000000*RTGS_latency):.10f} W")