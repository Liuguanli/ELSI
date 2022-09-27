#include "Constants.h"
// #include "Config.h"

const string Constants::RECORDS = "./files/records/";
const string Constants::QUERYPROFILES = "./files/queryprofile/";
const string Constants::DATASETS = "/home/research/datasets/";
const string Constants::DEFAULT_DISTRIBUTION = "skewed";
const string Constants::BUILD = "build/";
const string Constants::UPDATE = "update/";
const string Constants::POINT = "point/";
const string Constants::WINDOW = "window/";
const string Constants::ACCWINDOW = "accwindow/";
const string Constants::KNN = "knn/";
const string Constants::ACCKNN = "accknn/";
const string Constants::INSERT = "insert/";
const string Constants::DELETE = "delete/";
const string Constants::INSERTPOINT = "insertPoint/";
const string Constants::INSERTWINDOW = "insertWindow/";
const string Constants::INSERTACCWINDOW = "insertAccWindow/";
const string Constants::INSERTKNN = "insertKnn/";
const string Constants::INSERTACCKNN = "insertAccKnn/";
const string Constants::LEARNED_CDF = "learned_cdf/";

const string Constants::PRE_TRAIN_1D_DATA = "./method_pool/MR/pre_train/1D_data/0.1/";
const string Constants::FEATURES_PATH_ZM = "./method_pool/MR/pre_train/features_zm/0.1/";
const string Constants::PRE_TRAIN_MODEL_PATH_ZM = "./method_pool/MR/pre_train/models_zm/0.1/";

const string Constants::PRE_TRAIN_2D_DATA = "./method_pool/MR/pre_train/2D_data/";
const string Constants::PRE_TRAIN_MODEL_PATH_RSMI = "./method_pool/MR/pre_train/models_rsmi/";
const string Constants::FEATURES_PATH_RSMI = "./method_pool/MR/pre_train/features_rsmi/";

const string Constants::DEFAULT_PRE_TRAIN_MODEL_PATH = "./method_pool/MR/pre_train/models_zm/0.1/index_0.pt";
const string Constants::DEFAULT_PRE_TRAIN_MODEL_PATH_RSMI = "./method_pool/MR/pre_train/models_rsmi/Z/skewed_1000_1_1_.pt";
const string Constants::DEFAULT_PRE_TRAIN_MODEL_PATH_RSMI_H = "./method_pool/MR/pre_train/models_rsmi/H/skewed_1000_1_1_.pt";

// const string Constants::CLUSTER_FILE = "/home/liuguanli/Dropbox/research/BASE/method_pool/CL/cluster.py";
const string Constants::CLUSTER_FILE = "./method_pool/CL/cluster.py";
const string Constants::RL_FILE = "./method_pool/RL/rl_4_sfc/RL_4_SFC.py";
const string Constants::RL_FILE_RSMI = "./method_pool/RL/rl_4_sfc/RL_4_SFC_RSMI.py";

const string Constants::BUILD_TIME_MODEL_PATH = "./data/build_time_model_zm.pt";
const string Constants::QUERY_TIME_MODEL_PATH = "./data/query_time_model_zm.pt";
const string Constants::RAW_DATA_PATH = "./data/scorer_raw_data.csv";

const string Constants::SYNTHETIC_DATA_PATH = "/home/research/datasets/BASE/synthetic/";

const string Constants::REBUILD_RAW_DATA_PATH = "./data/rebuild_raw_data.csv";
const string Constants::REBUILD_DATA_PATH = "./data/rebuild_set_formatted.csv";
const string Constants::REBUILD_MODEL_PATH = "./data/rebuild_model.pt";

const double Constants::LEARNING_RATE = 0.05;

Constants::Constants()
{
}