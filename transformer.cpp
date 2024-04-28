//
// Created by 80324821 on 2023/4/19.
//

#include "transformer.h"
#include "tensor.h"
#include <math.h>

struct MultiHeadAttention* rand_init_MultiHeadAttention(int max_len,int n_heads,int d_k,int d_model){
    struct MultiHeadAttention* mha = (struct MultiHeadAttention *)malloc(sizeof(struct MultiHeadAttention));
    mha->max_len=max_len;
    mha->n_heads=n_heads;
    mha->d_k=d_k;
    mha->d_model=d_model;

    mha->W_Q = rand_tensor(mha->d_model,mha->n_heads*mha->d_k);
    mha->W_K = rand_tensor(mha->d_model,mha->n_heads*mha->d_k);
    mha->W_V = rand_tensor(mha->d_model,mha->n_heads*mha->d_k);
    mha->linear = rand_tensor(mha->n_heads*mha->d_k,mha->d_model);
    return mha;
};
/*
推理时batch_size为1 因此input是 1*len_seq*d_model 的tensor
而W_Q是一个(d_model,d_k*n_heads)的tensor
相乘之后是(len_seq,d_k*n_heads)的tensor
然后变换形状 然后分头 就是将n_heads换到第二个维度
变为(batch_size,n_heads,len_seq,d_k) tensor
这个时候len_seq换到倒数第二维 因此最后两维包含了整个seq的信息
这时候再将q_s和k_s做点积 最后两维是(len_seq*d_k)*(d_k*len_seq)=(len_seq*len_seq)
这个即是序列的attention score矩阵 表示每个token相互的关系
然后再乘以q_v即的到了一层encoder layer的输出
*/
struct tensor* computeMultiHeadAttention(tensor* input,struct MultiHeadAttention * attn_net){
    tensor* q_s = mm2d(input,attn_net->W_Q);
    view(q_s,attn_net->max_len,attn_net->n_heads,attn_net->d_k);
    q_s = transpose(q_s,0,1);

    tensor* k_s = mm2d(input,attn_net->W_K);
    view(k_s,attn_net->max_len,attn_net->n_heads,attn_net->d_k);
    k_s = transpose(k_s,0,1);

    tensor* v_s = mm2d(input,attn_net->W_V);
    view(v_s,attn_net->max_len,attn_net->n_heads,attn_net->d_k);
    v_s = transpose(v_s,0,1);

    tensor* scores = tensor_scaled_division(mm3d(q_s,transpose(k_s,1,2)),sqrt(attn_net->d_k));
    softmax_last_dim(scores);

    tensor* context = mm3d(scores,v_s);
    context = view(transpose(context,0,1),attn_net->max_len,attn_net->n_heads * attn_net->d_k);

    tensor* output = layer_norm(add(mm2d(context,attn_net->linear),input));
    return output;
}

struct tensor* computeCrossMultiHeadAttention(tensor* input,tensor* condition,struct MultiHeadAttention * attn_net){
    tensor* q_s = mm2d(input,attn_net->W_Q);
    view(q_s,attn_net->max_len,attn_net->n_heads,attn_net->d_k);
    q_s = transpose(q_s,0,1);

    tensor* k_s = mm2d(condition,attn_net->W_K);
    view(k_s,attn_net->max_len,attn_net->n_heads,attn_net->d_k);
    k_s = transpose(k_s,0,1);

    tensor* v_s = mm2d(condition,attn_net->W_V);
    view(v_s,attn_net->max_len,attn_net->n_heads,attn_net->d_k);
    v_s = transpose(v_s,0,1);

    tensor* scores = tensor_scaled_division(mm3d(q_s,transpose(k_s,1,2)),sqrt(attn_net->d_k));
    softmax_last_dim(scores);

    tensor* context = mm3d(scores,v_s);
    context = view(transpose(context,0,1),attn_net->max_len,attn_net->n_heads * attn_net->d_k);

    tensor* output = layer_norm(add(mm2d(context,attn_net->linear),input));
    return output;
}

struct Transformer {
    // 编码器和解码器层数
    int num_layers;
    // 每个编码器和解码器的多头自注意力头数
    int num_heads;
    // 隐藏层维度
    int hidden_size;
    // 输入序列长度
    int max_seq_length;
    // 词表大小
    int vocab_size;
//    // 位置编码矩阵
//    float position_embedding[max_seq_length][hidden_size];
//    // 词向量矩阵
//    float embedding[vocab_size][hidden_size];
//    // 编码器层
//    struct EncoderLayer encoder_layer[num_layers];
//    // 解码器层
//    struct DecoderLayer decoder_layer[num_layers];
};

void MultiHeadAttentionTest(){
    struct MultiHeadAttention* mha = rand_init_MultiHeadAttention(16,2,32,64);
    struct tensor* input = rand_tensor(16,64);
    tensor* out = computeMultiHeadAttention(input,mha);
    tensor_print(out);
}

