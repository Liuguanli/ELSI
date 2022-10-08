import matplotlib.pyplot as plt
from dqn import DeepQNetwork
from ddpg import DeepDeterministicPolicyGradient
import numpy as np
import sys, getopt
import copy
import pandas as pd
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True  # this works to disable the WARNINGS

def cal_dist(source_cdf, target_cdf):
    max_dist = 0.0
    length = len(source_cdf)
    for i in range(length):
        temp_dist = abs(source_cdf[i] - target_cdf[i])
        if max_dist < temp_dist:
            max_dist = temp_dist
    return max_dist

def cal_reward(source_cdf, target_cdf):
    length = len(source_cdf)
    gap = int(len(target_cdf) / length)
    max_err = 0.0
    for i in range(length):
        if abs(source_cdf[i] - target_cdf[i * gap]) > max_err:
            max_err = abs(source_cdf[i] - target_cdf[i * gap])
        
    return max_err
    # length = len(source_cdf)
    # res = 0.0
    # for i in range(length):
    #     res += abs(source_cdf[i] - target_cdf[i])
    # return res / length

def init_sfc(length):
    pdf = [1.0 / length for i in range(length)]
    cdf = []
    num = 0.0
    sfc = [1 for i in range(length)]
    for item in pdf:
        num += item
        cdf.append(num)
    return pdf, cdf, sfc

def get_pdf_cdf_from_sfc(sfc, print_gap = False):
    length = sum(sfc)
    pdf = [i * 1.0 / length for i in sfc]
    cdf = []
    num = 0.0
    for item in pdf:
        num += item
        cdf.append(num)
    # TODO change cdf
    start_index = 0
    start = cdf[start_index]
    for i in range(1, len(cdf)):
        if cdf[i] == start:
            continue
        else:
            if (i - start_index) > 1:
                gap = (cdf[i] - start) / (i - start_index)
                if print_gap:
                    print("start_index",start_index)
                    print("cdf[start_index]",cdf[start_index])
                    print("i",i)
                    print("cdf[i]",cdf[i])
                for j in range(1, i - start_index):
                    # print("before cdf[start_index + j]", cdf[start_index + j])
                    cdf[start_index + j] += gap * j
                    # print("after cdf[start_index + j]", cdf[start_index + j])
            start_index = i
            start = cdf[start_index]
    return pdf, cdf

def get_pdf_cdf(path):
    pdf = []
    cdf = []
    with open(path, "r") as f:
        for line in f:
            cols = line.strip().split(",")
            pdf_item = float(cols[0])
            cdf_item = float(cols[1])
            cdf.append(cdf_item)
            pdf.append(pdf_item)
    return pdf, cdf

def choose_RL(name, length):
    if name == 'dqn':
        RL = DeepQNetwork(length, length)
    elif name == 'ddpg':
        REPLACEMENT = [
                dict(name='soft', tau=0.01),
                dict(name='hard', rep_iter_a=600, rep_iter_c=500)
            ][0]
        LR_A=0.01
        LR_C=0.01
        GAMMA=0.9
        EPSILON=0.1
        VAR_DECAY=.9995
        RL = DeepDeterministicPolicyGradient(length, length, 1, LR_A, LR_C, REPLACEMENT, GAMMA,
                                                EPSILON)
    return RL

def train_sfc(RL, sfc, cdf, target_cdf):
    iteration = 50000
    MEMORY_CAPACITY=10000
    step = 0
    min_dist = 1.0
    min_sfc = []
    while True:
        source_pdf, source_cdf = get_pdf_cdf_from_sfc(sfc)
        action = RL.choose_action(np.array(sfc))
        sfc1 = copy.deepcopy(sfc)
        sfc1[action] = 0
        pdf1, cdf1 = get_pdf_cdf_from_sfc(sfc1)
        sfc2 = copy.deepcopy(sfc)
        sfc2[action] = 1
        pdf2, cdf2 = get_pdf_cdf_from_sfc(sfc2)
        dist = cal_reward(source_cdf, target_cdf)
        dist1 = cal_reward(cdf1, target_cdf)
        dist2 = cal_reward(cdf2, target_cdf)

        if sfc[action] == 0:
            if dist1 < dist2:
                reward = -0.1 # TODO change this -0.001
                sfc_new = sfc1
            else:
                reward = dist1 - dist2
                sfc_new = sfc2
            RL.store_transition(np.array(sfc), action, reward, np.array(sfc2))
            reward = 0
            RL.store_transition(np.array(sfc), action, reward, np.array(sfc1))
        else:
            if dist1 < dist2:
                reward = dist2 - dist1
                sfc_new = sfc1
            else:
                reward = -0.1
                sfc_new = sfc2
            RL.store_transition(np.array(sfc), action, reward, np.array(sfc1))
            reward = 0
            RL.store_transition(np.array(sfc), action, reward, np.array(sfc2))
        if (step > MEMORY_CAPACITY) and (step % 5 == 0):
            RL.learn()
        temp_dist = cal_reward(source_cdf, target_cdf)
        if temp_dist < min_dist:
            min_dist = temp_dist
            min_sfc = sfc_new
        if step > iteration:
            # print(temp_dist)
            break
        sfc = sfc_new
        step += 1
    # print(min_dist)
    return min_sfc, min_dist
    # return sfc

def load_data(name):
    data = pd.read_csv(name, header=None)
    x = data[0].values.reshape(-1, 1)
    y = data[1].values.reshape(-1, 1)
    return x, y

def draw_cdf_8t8(source_cdfs, new_cdfs, target_cdfs, labels):
    col_num = len(source_cdfs)
    for i in range(col_num):
        length = len(source_cdfs[i])
        x = [(i) * 1.0 / (length-1) for i in range(length)]
        plt.plot(x, new_cdfs[i], label="real")
        # plt.title(labels[i])
    resolution = 54
    x, y = load_data('/home/liuguanli/Documents/pre_train/features_zm/' + str(resolution) + '_OSM_100000000_1_2_.csv')
    side = int(pow(2, resolution / 2))
    plt.plot(x, y, label="synthetic")

    plt.legend(fontsize=20, loc="best", ncol=1)

    plt.savefig("sfcs.png", format='png', bbox_inches='tight')
    plt.savefig("sfcs.eps", format='eps', bbox_inches='tight')
    plt.show()

def draw_cdf(source_cdfs, new_cdfs, target_cdfs, labels):
    col_num = len(source_cdfs)
    for i in range(col_num):
        length = len(source_cdfs[i])
        x = [(i+1) * 1.0 / length for i in range(length)]
        
        plt.subplot(col_num, 3, i * 3 + 1)
        plt.plot(x, source_cdfs[i])
        

        plt.subplot(col_num, 3, i * 3 + 2)
        plt.plot(x, new_cdfs[i])

        plt.title(labels[i])

        plt.subplot(col_num, 3, i * 3 + 3)
        plt.plot(x, target_cdfs[i])
    plt.subplots_adjust(hspace=1.5, wspace=0.5)
    plt.savefig("sfcs.png", format='png', bbox_inches='tight')
    plt.show()

def run(bin_num, target_cdf, method_name):
    sfc = [1 for i in range(bin_num)]
    source_cdf = [float(i+1)/bin_num for i in range(bin_num)]
    new_sfc, min_dist = train_sfc(choose_RL(method_name, bin_num), sfc, source_cdf, target_cdf)
    new_pdf, new_cdf = get_pdf_cdf_from_sfc(new_sfc)
    return source_cdf, new_cdf, target_cdf, min_dist, new_sfc


def write_SFC(name, bit_num, sfc, cdf):
    all_fo = open("/home/liuguanli/Documents/pre_train/sfc_z/" + name, "w")
    num = len(sfc)
    for i in range(num):
        all_fo.write(str(sfc[i]) + "," + str(cdf[i]) + "\n")
    all_fo.close()

def get_locations_via_zvalue(bit_num, z_value):
    x = 0
    y = 0
    for i in range(bit_num):
        xSeed = 1 << (2 * i + 0)
        ySeed = 1 << (2 * i + 1)

        if z_value & xSeed > 0:
            x += pow(2, i)
        if z_value & ySeed > 0:
            y += pow(2, i)
    return x, y

def write_SFC_2d(name, bit_num, sfc, cdf):
    all_fo = open(name, "w")
    num = len(sfc)
    side = pow(2, bit_num)
    gap = 1.0 / side
    points_num = 10
    for i in range(num):
        if sfc[i] == 0:
            continue
        x, y = get_locations_via_zvalue(bit_num, i)
        for j in range(points_num):
            all_fo.write(str(random.uniform(x * gap, (x + 1) * gap)) + "," + str(random.uniform(y * gap, (y + 1) * gap)) + "," + str(cdf[i]) + "\n")
    all_fo.close()

def parser(argv):
    try:
        opts, args = getopt.getopt(argv, "i:o:b:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            input_file = arg
        elif opt == '-o':
            output_file = arg
        elif opt == '-b':
            bin_num = int(arg)

    return input_file, output_file, bin_num

if __name__ == "__main__":
    input_file, output_file, bin_num = parser(sys.argv[1:])

    source_cdfs = []
    new_cdfs = []
    target_cdfs = []
    labels = []
    # bit_nums = [6, 8, 10]
    # bit_nums = [6]
    # method_names = ['dqn', 'ddpg']
    method_names = ['dqn']
    # for bit_num in bit_nums:
    for method_name in method_names:

        target_pdf, target_cdf = get_pdf_cdf(input_file)
        source_cdf, new_cdf, target_cdf, min_dist, new_sfc = run(bin_num, target_cdf, method_name)
        # print(target_cdf)
        # print(new_cdf)

        bit_num_map={64:3, 256:4, 1024:5}
        write_SFC_2d(output_file, bit_num_map[bin_num], new_sfc, new_cdf)
        source_cdfs.append(source_cdf)
        new_cdfs.append(new_cdf)
        target_cdfs.append(target_cdf)
        labels.append(method_name + "-" + str(bin_num) + " cells    dist=" + str(min_dist))
    # draw_cdf(source_cdfs, new_cdfs, target_cdfs, labels)
    # draw_cdf_8t8(source_cdfs, new_cdfs, target_cdfs, labels)
