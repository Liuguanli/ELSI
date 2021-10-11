#ifndef EXTRASTORAGEBLOCK_H
#define EXTRASTORAGEBLOCK_H

#include <vector>
#include "Point.h"
#include "Mbr.h"
#include "../utils/Constants.h"
using namespace std;

template <typename D>

class ExtraStorageBlock
{
public:
    int capacity = Constants::PAGESIZE;
    vector<D> children;
    ExtraStorageBlock<D>() {}
    void add_point(D point) { children.push_back(point); }
    void add_points(vector<D> points){children.insert(children.end(), points.front(), points.end());}
    bool delete_point(D point)
    {
        return true;
    }
    bool is_full() { return children.size() == capacity; }
    // ExtraStorageBlock split();
};

#endif