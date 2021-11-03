#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
using namespace std;
class Constants
{
public:
    // status
    static const int STATUS_FRAMEWORK_INIT = 0;
    static const int STATUS_FRAMEWORK_INIT_DONE = 1;
    static const int STATUS_FRAMEWORK_INIT_BUILD_PROCESSOR = 2;
    static const int STATUS_FRAMEWORK_INIT_BUILD_PROCESSOR_LOAD_DONE = 3;
    static const int STATUS_FRAMEWORK_INIT_BUILD_PROCESSOR_TRAIN_DONE = 4;

    static const int STATUS_FRAMEWORK_INIT_REBUILD_PROCESSOR = 5;
    static const int STATUS_FRAMEWORK_INIT_REBUILD_PROCESSOR_LOAD_DONE = 6;
    static const int STATUS_FRAMEWORK_INIT_REBUILD_PROCESSOR_TRAIN_DONE = 7;

    static const int STATUS_FRAMEWORK_BUILD = 8;
    static const int STATUS_FRAMEWORK_BUILD_DONE = 9;

    static const int STATUS_FRAMEWORK_BEGIN_QUERY = 10;
    static const int STATUS_FRAMEWORK_BEGIN_QUERY_DONE = 11;

    static const int DEFAULT_BIN_NUM = 100;
    static const int CLUSTER_K = 100;

    static const int CL = 0;
    static const int MR = 1;
    static const int OG = 2;
    static const int RL = 3;
    static const int RS = 4;
    static const int SP = 5;

    static const int QUERY_TYPE_POINT = 1;
    static const int QUERY_TYPE_WINDOW = 2;
    static const int QUERY_TYPE_KNN = 3;

    static const int DIM = 2;
    static const int PAGESIZE = 100;
    static const int EACH_DIM_LENGTH = 8;
    static const int INFO_LENGTH = 8;
    static const int MAX_WIDTH = 16;
    static const int EPOCH = 500;
    static const int START_EPOCH = 300;
    static const int EPOCH_ADDED = 100;
    static const int BIT_NUM = 6;
    static const int SCORER_WIDTH = 32;

    static const int THRESHOLD = 20000;

    static const int DEFAULT_SIZE = 16000000;
    static const int DEFAULT_SKEWNESS = 4;

    static const int UNIFIED_Z_BIT_NUM = 4;
    // static const int UNIFIED_Z_BIT_NUM  = 6;

    static const int UNIFIED_H_BIT_NUM = 4;
    // static const int UNIFIED_H_BIT_NUM  = 10;

    static const int RESOLUTION = 1;

    // static const int HIDDEN_LAYER_WIDTH = 16;
    // static const bool IS_MODEL_REUSE = true;
    // static const bool IS_SAMPLING = false;
    // static const bool IS_RL_SFC = true;

    // static const bool IS_SAMPLING = false;
    // static const int HIDDEN_LAYER_WIDTH = 50;
    // static const bool IS_MODEL_REUSE = false;
    // static const bool IS_RL_SFC = false;

    // static const bool IS_SAMPLING_FIRST = false;

    // static const bool IS_SAMPLING = false;
    // static const bool IS_MODEL_REUSE = false;
    static const int HIDDEN_LAYER_WIDTH = 8;
    // static const bool IS_RL_SFC = false;

    // static const bool IS_CLUSTER = true;
    // static const bool IS_REPRESENTATIVE_SET = false;
    // static const bool IS_CLUSTER = false;
    // static const bool IS_REPRESENTATIVE_SET = true;
    // static const bool IS_REPRESENTATIVE_SET_SPACE = true;
    // static const bool IS_RL_SFC = false;
    // static const double MODEL_REUSE_THRESHOLD;

    static const double SAMPLING_RATE;
    static const double LEARNING_RATE;
    static const string RECORDS;
    static const string QUERYPROFILES;
    static const string DATASETS;

    static const string DEFAULT_DISTRIBUTION;

    static const string BUILD;
    static const string UPDATE;
    static const string POINT;
    static const string WINDOW;
    static const string ACCWINDOW;
    static const string KNN;
    static const string ACCKNN;
    static const string INSERT;
    static const string DELETE;
    static const string INSERTPOINT;
    static const string INSERTWINDOW;
    static const string INSERTACCWINDOW;
    static const string INSERTKNN;
    static const string INSERTACCKNN;
    static const string DELETEPOINT;
    static const string DELETEWINDOW;
    static const string DELETEACCWINDOW;
    static const string DELETEKNN;
    static const string DELETEACCKNN;

    static const string LEARNED_CDF;

    static const string PRE_TRAIN_1D_DATA;

    static const string FEATURES_PATH_ZM;
    static const string PRE_TRAIN_MODEL_PATH_ZM;
    static const string PRE_TRAIN_MODEL_PATH_RSMI;
    static const string PRE_TRAIN_2D_DATA;

    static const string DEFAULT_PRE_TRAIN_MODEL_PATH;
    static const string DEFAULT_PRE_TRAIN_MODEL_PATH_RSMI;
    static const string DEFAULT_PRE_TRAIN_MODEL_PATH_RSMI_H;

    static const string FEATURES_PATH_RSMI;

    static const string CLUSTER_FILE;
    static const string RL_FILE;
    static const string RL_FILE_RSMI;

    static const string BUILD_TIME_MODEL_PATH;
    static const string QUERY_TIME_MODEL_PATH;
    static const string RAW_DATA_PATH;

    static const string REBUILD_RAW_DATA_PATH;
    static const string REBUILD_DATA_PATH;
    static const string REBUILD_MODEL_PATH;

    static const string SYNTHETIC_DATA_PATH;

    Constants();
};

#endif