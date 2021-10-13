#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>

using namespace std;

template <typename D, typename T>
class DataSet
{
    // TODO need a method to transform points to keys and labels.

public:
    // double max_x = numeric_limits<double>::min();
    // double max_y = numeric_limits<double>::min();
    // double min_x = numeric_limits<double>::max();
    // double min_y = numeric_limits<double>::max();

    double max_x = 1;
    double max_y = 1;
    double min_x = 0;
    double min_y = 0;

    vector<D> points;
    vector<T> keys;
    vector<float> normalized_keys;
    vector<float> labels;

    string dataset_name;

    inline static vector<D> (*read_data_pointer)(string, string, double &, double &, double &, double &);
    inline static void (*mapping_pointer)(vector<D> &, vector<T> &);
    inline static void (*save_data_pointer)(vector<D>, string);

    // void generate_keys(vector<T> (*pfun)(vector<D>))
    // {
    //     generate_keys_pointer = pfun;
    //     keys = generate_keys_pointer(points);
    //     generate_labels();
    // }

    void mapping()
    {
        mapping_pointer(points, keys);
        generate_labels();
        generate_normalized_keys();
    }

    void generate_normalized_keys()
    {
        int N = keys.size();
        T first_key = keys[0];
        T last_key = keys[keys.size() - 1];
        T gap = last_key - first_key;
        for (size_t i = 0; i < N; i++)
        {
            normalized_keys.push_back((float)(keys[i] - first_key) / gap);
        }
    }

    void generate_labels()
    {
        int N = keys.size();
        for (size_t i = 0; i < N; i++)
        {
            labels.push_back((float)i / N);
        }
    }

    void read_data()
    {
        // assert(dataset_name != NULL && dataset_name != "");
        points = read_data_pointer(dataset_name, ",", min_x, min_y, max_x, max_y);
    }

    void read_cdf(string file_name)
    {
        ifstream file(file_name);
        string line = "";
        while (getline(file, line))
        {
            vector<string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(","));
            normalized_keys.push_back(stof(vec[1]));
        }

        file.close();
        int N = normalized_keys.size();
        for (size_t i = 0; i < N; i++)
        {
            labels.push_back((float)i / N);
        }
    }

    // void read_data(vector<D> (*pfun)(string, string, double &, double &, double &, double &))
    // {
    //     // assert(dataset_name != NULL && dataset_name != "");
    //     read_data_pointer = pfun;
    //     points = read_data_pointer(dataset_name, ",", min_x, min_y, max_x, max_y);
    // }

    string path = "./data/temp.csv";
    void save_temp_data()
    {
        cout << "points.size() : " << points.size() << endl;
        save_data_pointer(points, path);
    }

    void remove_temp_data()
    {
        char s[path.length() + 1];
        strcpy(s, path.c_str());
        remove(s);
    }

    DataSet() {}
    DataSet(string path) { this->dataset_name = path; }
    DataSet(vector<D> &points)
    {
        this->points = points;
        for (size_t i = 0; i < points.size(); i++)
        {
            keys.push_back(points[i].curve_val);
            labels.push_back(points[i].label);
            normalized_keys.push_back(points[i].normalized_key);
        }
    }

    DataSet(vector<T> &keys, vector<float> &labels)
    {
        this->labels = labels;
        this->keys = keys;
        generate_normalized_keys();
    }
};

#endif