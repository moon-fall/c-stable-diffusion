//
// Created by 80324821 on 2023/4/19.
//

#ifndef UNTITLED_TRANSFORMER_H
#define UNTITLED_TRANSFORMER_H

#include "tensor.h"

struct MultiHeadAttention{
    int max_len;
    int d_model;
    int n_heads;
    int d_k;

    tensor* W_Q;
    tensor* W_K;
    tensor* W_V;

    tensor* linear;
};

struct MultiHeadAttention* rand_init_MultiHeadAttention(int max_len,int n_heads,int d_k,int d_model);

class transformer {

};


#endif //UNTITLED_TRANSFORMER_H
