#ifndef BUCKET_H
#define BUCKET_H

#include <iostream>
#include <vector>
#include <queue>
#include "Point.h"
#include "Mbr.h"
#include <typeinfo>
#include "../curves/hilbert.H"
#include "../curves/hilbert4.H"
#include <map>
#include "LeafNode.h"
#include <boost/smart_ptr/make_shared_object.hpp>
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"

class Bucket
{

private:
    // void addNode(LeafNode *);
    long long N = 0;
    int learned_dim = 0;

    Mbr mbr;
    std::shared_ptr<MLP> mlp;

    vector<Point> all_points;
    vector<LeafNode> leafnodes;
    int leaf_node_num = 0;
    int page_size = Constants::PAGESIZE;
    int max_error = 0;
    int min_error = 0;

public:
    Bucket(int learned_dim, Mbr mbr)
    {
        this->mbr = mbr;
        this->learned_dim = learned_dim;
    }

    Bucket(int learned_dim)
    {
        this->learned_dim = learned_dim;
    }

    void init(vector<Point> &points, vector<float> &keys, vector<float> &labels)
    {
        N = points.size();
        if (learned_dim == 0)
        {
            for (size_t i = 0; i < N; i++)
            {
                sort(points.begin(), points.end(), sortX());
                keys.push_back(points[i].x);
                labels.push_back(i * 1.0 / N);
            }
        }
        else
        {
            for (size_t i = 0; i < N; i++)
            {
                sort(points.begin(), points.end(), sortY());
                keys.push_back(points[i].y);
                labels.push_back(i * 1.0 / N);
            }
        }
        leaf_node_num = points.size() / page_size;
        for (int i = 0; i < leaf_node_num; i++)
        {
            LeafNode leafNode;
            auto bn = points.begin() + i * page_size;
            auto en = points.begin() + i * page_size + page_size;
            vector<Point> vec(bn, en);
            leafNode.add_points(vec);
            leafnodes.push_back(leafNode);
        }

        if (points.size() > page_size * leaf_node_num)
        {
            LeafNode leafNode;
            auto bn = points.begin() + page_size * leaf_node_num;
            auto en = points.end();
            vector<Point> vec(bn, en);
            leafNode.add_points(vec);
            leafnodes.push_back(leafNode);
            leaf_node_num++;
        }
    }

    void build(ExpRecorder &exp_recorder, vector<Point> &points, ELSI<Point, long long> &framework)
    {

        DataSet<Point, long long> original_data_set;
        vector<float> labels;
        vector<float> keys;
        init(points, keys, labels);
        original_data_set.points = points;
        original_data_set.normalized_keys = keys;
        original_data_set.labels = labels;
        int method = exp_recorder.build_method;
        method = Constants::OG;
        std::shared_ptr<MLP>
            mlp_ = framework.get_build_method(original_data_set, method);

        for (size_t i = 0; i < N; i++)
        {
            float key = learned_dim == 0 ? points[i].x : points[i].y;
            int predicted_index = (int)(mlp_->predict_ZM(key) * leaf_node_num);

            predicted_index = max(predicted_index, 0);
            predicted_index = min(predicted_index, leaf_node_num - 1);

            int error = i / page_size - predicted_index;
            max_error = max(max_error, error);
            min_error = min(min_error, error);
        }
        // print("min_error: " + str(min_error) + " max_error: " + str(max_error));

        mlp_->max_error = max_error;
        mlp_->min_error = min_error;
        mlp = mlp_;
    }

    bool point_query(Point point)
    {
        float key = learned_dim == 0 ? point.x : point.y;
        int predicted_index = mlp->predict_ZM(key) * leaf_node_num;
        predicted_index = max(predicted_index, 0);
        predicted_index = min(predicted_index, leaf_node_num - 1);
        int front = predicted_index + mlp->min_error;
        front = max(front, 0);
        int back = predicted_index + mlp->max_error;
        back = min(back, leaf_node_num - 1);
        while (front <= back)
        {
            int mid = (front + back) / 2;
            LeafNode temp = leafnodes[mid];
            float first_key = learned_dim == 0 ? temp.children[0].x : temp.children[0].y;
            float last_key = learned_dim == 0 ? temp.children[temp.children.size() - 1].x : temp.children[temp.children.size() - 1].y;
            if (first_key <= key && key <= last_key)
            {
                vector<Point>::iterator iter = find(leafnodes[mid].children.begin(), leafnodes[mid].children.end(), point);
                if (iter != leafnodes[mid].children.end())
                {
                    return true;
                }
                else
                {
                    if (first_key == key)
                    {
                        back = mid - 1;
                    }
                    if (key == last_key)
                    {
                        front = mid + 1;
                    }
                }
            }
            else
            {
                if (first_key < key)
                {
                    front = mid + 1;
                }
                else
                {
                    back = mid - 1;
                }
            }
        }
        return false;
    }

    void window_query(Mbr query_window, vector<Point>& results)
    {
        float from = learned_dim == 0 ? query_window.y1 : query_window.x1;
        float to = learned_dim == 0 ? query_window.y2 : query_window.x2;
        float mbr_from = learned_dim == 0 ? mbr.y1 : mbr.x1;
        float mbr_to = learned_dim == 0 ? mbr.y2 : mbr.x2;

        float left_key = learned_dim == 0 ? query_window.x1 : query_window.y1;
        float right_key = learned_dim == 0 ? query_window.x2 : query_window.y2;



        if (from <= mbr_from && to >= mbr_to)
        {

        }
        else
        {

        }

    }

    void knn_query(int k, Point &query_point)
    {
    }

    bool insert(Point point)
    {

        // dynArray.push_back(point);

        // bool isSplit = false;
        // bool isAdded = false;
        // for (size_t i = 0; i < LeafNodes.size(); i++)
        // {
        //     if (LeafNodes[i].mbr.contains(point))
        //     {
        //         LeafNodes[i].children->push_back(point);
        //         isAdded = true;
        //         if (LeafNodes[i].children->size() > Constants::PAGESIZE)
        //         {
        //             Mbr oldMbr = LeafNodes[i].mbr;
        //             // new LeafNode
        //             LeafNode newSplit;
        //             int mid = 0;
        //             if (splitX)
        //             {
        //                 sort(LeafNodes[i].children->begin(), LeafNodes[i].children->end(), sortX());
        //                 float midX = (oldMbr.x1 + oldMbr.x2) / 2;
        //                 Mbr mbr(midX, oldMbr.y1, oldMbr.x2, oldMbr.y2);
        //                 oldMbr.x2 = midX;
        //                 newSplit.mbr = mbr;
        //                 isSplit = true;
        //                 for (size_t j = 0; j < LeafNodes[i].children->size(); j++)
        //                 {
        //                     if ((*(LeafNodes[i].children))[j].x > midX)
        //                     {
        //                         mid = j;
        //                         break;
        //                     }
        //                 }
        //             }
        //             else
        //             {
        //                 sort(LeafNodes[i].children->begin(), LeafNodes[i].children->end(), sortY());
        //                 float midY = (oldMbr.y1 + oldMbr.y2) / 2;
        //                 Mbr mbr(oldMbr.x1, midY, oldMbr.x2, oldMbr.y2);
        //                 oldMbr.y2 = midY;
        //                 newSplit.mbr = mbr;
        //                 isSplit = true;
        //                 for (size_t j = 0; j < LeafNodes[i].children->size(); j++)
        //                 {
        //                     if ((*(LeafNodes[i].children))[j].y > midY)
        //                     {
        //                         mid = j;
        //                         break;
        //                     }
        //                 }
        //             }

        //             vector<Point> vec(LeafNodes[i].children->begin() + mid, LeafNodes[i].children->end());
        //             newSplit.children->insert(newSplit.children->end(), vec.begin(), vec.end());
        //             // cout<< "newSplit->children size: " << newSplit->children->size() << endl;
        //             // old LeafNode
        //             vector<Point> vec1(LeafNodes[i].children->begin(), LeafNodes[i].children->begin() + mid);
        //             LeafNodes[i].children->clear();
        //             LeafNodes[i].children->insert(LeafNodes[i].children->end(), vec1.begin(), vec1.end());
        //             // cout<< "LeafNodes[i]->children size: " << LeafNodes[i]->children->size() << endl;
        //             LeafNodes.insert(LeafNodes.begin() + i, newSplit);

        //             splitX = !splitX;
        //         }
        //         break;
        //     }
        // }

        // return isSplit;
    }

    vector<Point> getAllPoints()
    {
        // for (LeafNode LeafNode : LeafNodes)
        // {
        //     vector<Point> *tempResult = LeafNode.children;
        //     result.insert(result.end(), tempResult->begin(), tempResult->end());
        // }
        return all_points;
    }

    void remove(Point point)
    {
        // for (LeafNode LeafNode : LeafNodes)
        // {
        //     if (LeafNode.mbr.contains(point) && LeafNode.delete_point(point))
        //     {
        //         // cout << "remove it" << endl;
        //         break;
        //     }
        // }
    }

    void addNode(LeafNode node)
    {
        // add
        leafnodes.push_back(node);
        // update MBR
        mbr.update(node.mbr);
    }
};
#endif