#ifndef METHOD_H
#define METHOD_H
#include "../utils/ModelTools.h"

namespace model_training
{
    template <typename K, typename L>
    std::shared_ptr<MLP> real_train(vector<K> keys, vector<L> labels)
    {
        int width = keys.size() / labels.size();
        auto mlp = std::make_shared<MLP>(width);

#ifdef use_gpu
        mlp->to(torch::kCUDA);
#endif
        mlp->train_model(keys, labels);
        if (width == 1)
        {
            mlp->get_parameters_ZM();
        }
        if (width == 2)
        {
            mlp->get_parameters();
        }
        return mlp;
    }

}

#endif