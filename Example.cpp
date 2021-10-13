#ifndef use_gpu
#define use_gpu
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <chrono>

#include "entities/Point.h"
#include "entities/Query.h"
#include "entities/DataSet.h"
#include "utils/FileReader.h"
#include "utils/FileWriter.h"
#include "utils/SortTools.h"
#include "curves/z.H"

#include "indices/ZM.h"
#include "ELSI.h"

using namespace std;
using namespace zm;


vector<Point> read_data(string filename, string delimeter, double &min_x, double &min_y, double &max_x, double &max_y)
{
    ifstream file(filename);

    vector<Point> points;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        Point point(stod(vec[0]), stod(vec[1]));

        max_x = max_x > point.x ? max_x : point.x;
        max_y = max_y > point.y ? max_y : point.y;
        min_x = min_x < point.x ? min_x : point.x;
        min_y = min_y < point.y ? min_y : point.y;

        points.push_back(point);
    }
    // Close the File
    file.close();

    return points;
}

vector<long> generate_keys(vector<Point> points)
{
    int N = points.size();
    int bit_num = ceil((log(N)) / log(2));
    for (long i = 0; i < N; i++)
    {
        long long xs[2] = {(long long)(points[i].x * N), (long long)(points[i].y * N)};
        points[i].curve_val = compute_Z_value(xs, 2, bit_num);
    }
    sort(points.begin(), points.end(), sort_curve_val());
    vector<long> keys(N);
    for (long i = 0; i < N; i++)
    {
        keys[i] = points[i].curve_val;
    }
    return keys;
}

int main(int argc, char **argv)
{
    // string dataset_name = "/home/research/datasets/skewed_2000000_4_2_.csv";
    string dataset_name = "/home/research/datasets/OSM_100000000_1_2_.csv";
    zm::init(dataset_name);
    zm::build_ZM();
    zm::query();

    // TODO record time
    // TODO write file
    // TODO RSMI
    // TODO lisa
    // TODO ML-index
}

#endif // use_gpu
