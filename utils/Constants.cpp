#include "Constants.h"

#ifdef __APPLE__
// const string Constants::RECORDS = "/Users/guanli/Dropbox/records/VLDB20/";
const string Constants::QUERYPROFILES = "/Users/guanli/Documents/datasets/RLRtree/queryprofile/";
// const string Constants::DATASETS = "/Users/guanli/Documents/datasets/RLRtree/raw/";
const string Constants::RECORDS = "./files/records/";
// const string Constants::QUERYPROFILES = "./files/queryprofile/";
const string Constants::DATASETS = "./datasets/";
#else
// const string Constants::RECORDS = "/home/liuguanli/Dropbox/records/VLDB20/";
// const string Constants::QUERYPROFILES = "./files/queryprofile/";
// const string Constants::DATASETS = "/home/liuguanli/Documents/datasets/RLRtree/raw/";
const string Constants::RECORDS = "./files/records/";
const string Constants::QUERYPROFILES = "./files/queryprofile/";
const string Constants::DATASETS = "/home/research/datasets/";
#endif
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
const string Constants::DELETEPOINT = "delete_point/";
const string Constants::DELETEWINDOW = "deleteWindow/";
const string Constants::DELETEACCWINDOW = "deleteAccWindow/";
const string Constants::DELETEKNN = "deleteKnn/";
const string Constants::DELETEACCKNN = "deleteAccKnn/";
const string Constants::LEARNED_CDF = "learned_cdf/";

const string Constants::TORCH_MODELS = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/torch_models/";
const string Constants::TORCH_MODELS_ZM = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/torch_models_zm/";

const string Constants::PRE_TRAIN_DATA = "/home/liuguanli/Documents/pre_train/2D_data/";

const string Constants::SYNTHETIC_SFC_Z = "/home/liuguanli/Documents/pre_train/sfc_z/";
const string Constants::SFC_Z_WEIGHT = "/home/liuguanli/Documents/pre_train/sfc_z_weight/";
const string Constants::SFC_Z_COUNT = "/home/liuguanli/Documents/pre_train/sfc_z_count/";

const string Constants::PRE_TRAIN_1D_DATA = "/home/liuguanli/Documents/pre_train/1D_data/0.1/";
const string Constants::FEATURES_PATH_ZM = "/home/liuguanli/Documents/pre_train/features_zm/1/0.1/";
const string Constants::PRE_TRAIN_MODEL_PATH_ZM = "/home/liuguanli/Documents/pre_train/models_zm/1/0.1/";

const string Constants::DEFAULT_PRE_TRAIN_MODEL_PATH = "/home/liuguanli/Documents/pre_train/models_zm/1/0.1/index_0.pt";

const string Constants::FEATURES_PATH_RSMI = "/home/liuguanli/Documents/pre_train/features_rsmi/";
const string Constants::PRE_TRAIN_MODEL_PATH_RSMI = "/home/liuguanli/Documents/pre_train/models_rsmi/";

const string Constants::CLUSTER_FILE = "/home/liuguanli/Dropbox/research/BASE/method_pool/CL/cluster.py";
const string Constants::RL_FILE = "/home/liuguanli/Dropbox/research/BASE/method_pool/RL/rl_4_sfc/RL_4_SFC.py";

const string Constants::BUILD_TIME_MODEL_PATH = "./data/build_time_model_zm.pt";
const string Constants::QUERY_TIME_MODEL_PATH = "./data/query_time_model_zm.pt";
const string Constants::RAW_DATA_PATH = "./data/scorer_raw_data.csv";

const string Constants::SYNTHETIC_DATA_PATH = "/home/research/datasets/BASE/synthetic/";

const string Constants::REBUILD_RAW_DATA_PATH = "./data/rebuild_raw_data.csv";
const string Constants::REBUILD_DATA_PATH = "./data/rebuild_set_formatted.csv";
const string Constants::REBUILD_MODEL_PATH = "./data/rebuild_model.pt";

const double Constants::LEARNING_RATE = 0.05;

// const double Constants::MODEL_REUSE_THRESHOLD = 0.1;
// const double Constants::SAMPLING_RATE = 0.01;

Constants::Constants()
{
}