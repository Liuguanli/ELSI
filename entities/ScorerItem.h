#ifndef SCORERITEM_H
#define SCORERITEM_H
#include <vector>
#include <string.h>
#include <string>
#include <math.h>
#include <sstream>
#include "../utils/Constants.h"

class ScorerItem
{

public:
    float dist = 0.0;
    float cardinality = 1.0;
    vector<float> method;
    float query_time;
    float build_time;

    ScorerItem() {}

    ScorerItem(float cardinality, float dist, vector<float> method)
    {
        this->cardinality = cardinality;
        this->dist = dist;
        this->method = method;
    }

    string get_ScorerItem()
    {
        stringstream ss;
        copy(method.begin(), method.end(), ostream_iterator<int>(ss, ","));
        string s = ss.str();
        s = s.substr(0, s.length() - 1);
        return to_string(cardinality) + "," + to_string(dist) + "," + s + "," + to_string(build_time) + "," + to_string(query_time);
    }
};

#endif
