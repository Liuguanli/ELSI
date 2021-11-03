#ifndef POINT_H
#define POINT_H
#include <vector>
// #include <string.h>
#include <string>
#include <math.h>

class Point
{

public:
    //  For ZM-index
    float x = 0.0;
    float y = 0.0;
    int x_i;
    int y_i;
    long long curve_val = 0;
    float label;
    int index = 0;
    float normalized_key;
    double temp_dist;

    // For ML-index and LISA
    int partition_id;
    double key;
    double normalized_ml_key;
    bool is_deleted = false;

    Point()
    {
    }

    Point(float x, float y)
    {
        this->x = x;
        this->y = y;
    }

    bool operator==(const Point &point)
    {
        if (this == &point)
        {
            return true;
        }
        else if (this->x == point.x && this->y == point.y)
        {
            return true;
        }
        return false;
    }

    // string get_self()
    // {
    //     return to_string(x) + "," + to_string(y) + "\n";
    // }

    double cal_dist(Point point)
    {
        temp_dist = sqrt(pow((point.x - x), 2) + pow((point.y - y), 2));
        return temp_dist;
    }
};

#endif
