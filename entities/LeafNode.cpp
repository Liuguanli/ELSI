#include <iostream>
#include "LeafNode.h"
#include "Point.h"
#include "../utils/Constants.h"
#include <algorithm>
using namespace std;

LeafNode::LeafNode() {}

LeafNode::LeafNode(Mbr mbr) { this->mbr = mbr; }

LeafNode::LeafNode(vector<Point> points) { add_points(points); }

void LeafNode::add_point(Point point)
{
    children.push_back(point);
    mbr.update(point.x, point.y);
}

void LeafNode::add_points(vector<Point> points)
{
    for (int i = 0; i < points.size(); i++)
    {
        add_point(points[i]);
    }
}

bool LeafNode::is_full()
{
    return children.size() >= Constants::PAGESIZE;
}

LeafNode LeafNode::split()
{
    // build rightNode
    LeafNode right;
    int mid = Constants::PAGESIZE / 2;
    vector<Point> vec(children.begin() + mid, children.end());
    right.add_points(vec);

    // build leftNode
    vector<Point> vec1(children.begin(), children.begin() + mid);
    children.clear();
    mbr.clean();
    add_points(vec1);
    return right;
}

bool LeafNode::delete_point(Point point)
{
    vector<Point>::iterator iter = find(children.begin(), children.end(), point);
    if (iter != children.end())
    {
        // cout << "find it" << endl;
        children.erase(iter);
        // update mbr
        if (!mbr.strict_contains(point))
        {
            mbr.clean();
            for (int i = 0; i < children.size(); i++)
            {
                mbr.update(point.x, point.y);
            }
        }
        return true;
    }
    return false;
}
