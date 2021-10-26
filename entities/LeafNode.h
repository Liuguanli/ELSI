#ifndef LEAFNODE_H
#define LEAFNODE_H

#include <vector>
#include "Point.h"
#include "Mbr.h"
using namespace std;

class LeafNode
{
public:
    int id;
    int level;
    Mbr mbr;
    vector<Point> children;
    LeafNode();
    LeafNode(int id);
    LeafNode(Mbr mbr);
    LeafNode(vector<Point>);
    void add_point(Point);
    void add_points(vector<Point>);
    bool delete_point(Point);
    bool is_full();
    LeafNode split();
};

#endif