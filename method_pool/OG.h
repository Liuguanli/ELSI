#ifndef OG_H
#define OG_H

#include "method.h"
#include <vector>
#include <string.h>
#include <string>
#include "../entities/DataSet.h"
#include "../utils/Constants.h"

template <typename D, typename T>
class OG
{

public:
    DataSet<D, T> do_og(DataSet<D, T> &dataset, int dimension)
    {
        if (dimension == 1)
        {
            return do_og(dataset);
        }
        if (dimension == 2)
        {
            return do_og_2d(dataset);
        }
        return dataset;
    }

    DataSet<D, T> do_og(DataSet<D, T> &dataset)
    {
        return dataset;
    }

    DataSet<D, T> do_og_2d(DataSet<D, T> &dataset)
    {
        DataSet<D, T> generated_dataset;
        return generated_dataset;
    }
};

#endif
