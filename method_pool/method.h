#ifndef METHOD_H
#define METHOD_H
#include "../utils/ModelTools.h"

namespace model_training
{
    template <typename K, typename L>
    std::shared_ptr<MLP> real_train_1d(vector<K> keys, vector<L> labels)
    {
        int width = keys.size() / labels.size();
        auto mlp = std::make_shared<MLP>(width);

#ifdef use_gpu
        mlp->to(torch::kCUDA);
#endif
        mlp->train_model(keys, labels);
        mlp->get_parameters_ZM();
        return mlp;
    }
}

#endif