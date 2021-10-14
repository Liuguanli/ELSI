#include "FileWriter.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include "util.h"
#include "../entities/Point.h"
#include "../entities/Mbr.h"

using namespace std;

FileWriter::FileWriter() {}

FileWriter::FileWriter(string filename)
{
    this->filename = filename;
    file_utils::check_dir(filename);
}

void FileWriter::write_statistics_items(vector<Statistics> items, string file_name)
{
    ofstream write;
    // file_utils::check_dir(filename);
    write.open(file_name, ios::out);
    int N = items.size();
    for (size_t i = 0; i < N; i++)
    {
        write << items[i].to_string() << endl;
    }
    write.close();
}

void FileWriter::write_score_items(vector<ScorerItem> items, string file_name)
{
    ofstream write;
    // file_utils::check_dir(file_name);
    write.open(file_name, ios::out);
    int N = items.size();
    for (size_t i = 0; i < N; i++)
    {
        write << items[i].get_ScorerItem() << endl;
    }
    write.close();
}

// void FileWriter::write_counted_SFC(vector<int> values, string name)
// {
//     ofstream write;
//     file_utils::check_dir(filename);
//     write.open((filename + name), ios::out);
//     int N = values.size();
//     for (size_t i = 0; i < N; i++)
//     {
//         write << values[i] << endl;
//     }
//     write.close();
// }

// void FileWriter::write_weighted_SFC(vector<float> values, string name)
// {
//     ofstream write;
//     file_utils::check_dir(filename);
//     write.open((filename + name), ios::out);
//     int N = values.size();
//     double sum = 0;
//     for (size_t i = 0; i < N; i++)
//     {
//         sum += (double)values[i];
//         write << values[i] << "," << sum << endl;
//     }
//     write.close();
// }

// void FileWriter::write_SFC(vector<float> values, string name)
// {
//     ofstream write;
//     file_utils::check_dir(filename);
//     write.open((filename + name), ios::out);
//     // cout<< "filename + name: " << filename + name << endl;
//     int N = values.size();
//     if (N > 1000000)
//     {
//         for (size_t i = 0; i < N; i++)
//         {
//             // if ((i + 1) % 100 == 0)
//             // {
//             write << values[i] << "," << (i + 1.0) / N << endl;
//             // }
//         }
//     }
//     else
//     {
//         for (size_t i = 0; i < N; i++)
//         {
//             write << values[i] << "," << (i + 1.0) / N << endl;
//         }
//     }
//     write.close();
// }

// void FileWriter::write_mbrs(vector<Mbr> mbrs, ExpRecorder exp_recorder)
// {
//     ofstream write;
//     string folder = Constants::WINDOW;
//     file_utils::check_dir(filename + folder);
//     write.open((filename + folder + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_" + to_string(exp_recorder.window_size) + "_" + to_string(exp_recorder.window_ratio) + ".csv"), ios::out);
//     for (Mbr mbr : mbrs)
//     {
//         write << mbr.get_self();
//     }
//     write.close();
// }

// void FileWriter::write_points(vector<Point> points,  string mid)
// {
//     ofstream write;
//     // string folder = Constants::KNN;
//     file_utils::check_dir(filename + mid);
//     cout << "path: " << filename + mid + exp_recorder.get_file_name() << endl;
//     write.open((filename + mid + exp_recorder.get_file_name()), ios::out);
//     for (Point point : points)
//     {
//         write << point.get_self();
//     }
//     write.close();
// }

void FileWriter::write_cost_model_data(int cardinality, string distribution, string method, double build_time, double query_time)
{
    ofstream write;
    file_utils::check_dir(filename);
    write.open((filename + "train_set.csv"), ios::app);
    write << cardinality << "," << distribution << "," << method << "," << build_time << "," << query_time << endl;
}

void FileWriter::write_build(ExpRecorder exp_recorder)
{
    ofstream write;
    string folder = Constants::RECORDS + Constants::BUILD;
    file_utils::check_dir(folder);
    write.open((folder + exp_recorder.get_file_name() + ".txt"), ios::app);
    write << exp_recorder.get_point_query_result();
    write.close();
}

void FileWriter::write_point_query(ExpRecorder exp_recorder)
{
    ofstream write;
    string folder = Constants::RECORDS + Constants::POINT;
    file_utils::check_dir(folder);
    write.open((folder + exp_recorder.get_file_name() + ".txt"), ios::app);
    write << exp_recorder.get_point_query_result();
    write.close();
}

void FileWriter::write_window_query(ExpRecorder exp_recorder)
{
    ofstream write;
    string folder = Constants::RECORDS + Constants::WINDOW;
    file_utils::check_dir(folder);
    write.open((folder + exp_recorder.get_file_name() + ".txt"), ios::app);
    write << exp_recorder.get_point_query_result();
    write.close();
}

void FileWriter::write_kNN_query(ExpRecorder exp_recorder)
{
    ofstream write;
    string folder = Constants::RECORDS + Constants::KNN;
    file_utils::check_dir(folder);
    write.open((folder + exp_recorder.get_file_name() + ".txt"), ios::app);
    write << exp_recorder.get_point_query_result();
    write.close();
}