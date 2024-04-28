//
// Created by 80324821 on 2023/4/21.
//

#ifndef UNTITLED_UNET_H
#define UNTITLED_UNET_H


#include <vector>
#include "tensor.h"

struct in_layers{
    struct tensor * group_norm_wight;//[320]/[640]/[1280]
    struct tensor * group_norm_bias;//[320]/[640]/[1280]

    struct tensor * conv_weight;//[320,320,3,3]/[640,640,3,3]/[1280,1280,3,3]
    struct tensor * conv_bias;//[320]/[640]/[1280]
};

struct emb_layers{
    struct tensor * linear_weight;//[320,1280]/[640,1280]/[1280,1280]
    struct tensor * linear_bias;//[320]/[640]/[1280]
};

struct out_layers{
    struct tensor * group_norm_wight;//[320]/[640]/[1280]
    struct tensor * group_norm_bias;//[320]/[640]/[1280]

    struct tensor * conv_weight;//[320,320,3,3]/[640,640,3,3]/[1280,1280,3,3]
    struct tensor * conv_bias;//[320]/[640]/[1280]
};

struct ResBlock{
    struct in_layers * in_layer;
    struct emb_layers * emb_layer;
    struct out_layers * out_layer;
    struct tensor * skip_connection_weight;//null/[640, 320, 1, 1]
    struct tensor * skip_connection_bias;
};

struct BasicTransformerBlock{
    struct CrossAttention * attn1;
    struct FeedForward * ff;
    struct CrossAttention * attn2;

    struct tensor * layer_norm_wight_1;//[320]
    struct tensor * layer_norm_bias_1;//[320]
    struct tensor * layer_norm_wight_2;//[320]
    struct tensor * layer_norm_bias_2;//[320]
    struct tensor * layer_norm_wight_3;//[320]
    struct tensor * layer_norm_bias_3;//[320]
};

struct SpatialTransformer{
    struct tensor * group_norm_weight;//[320]/[640]/[1280]
    struct tensor * group_norm_bias;//[320]/[640]/[1280]

    struct tensor * project_in_weight;//[320,320,1,1]/[640,640,1,1]/[1280,1280,1,1]   padding=0
    struct tensor * project_in_bias;//[320]/[640]/[1280]

    BasicTransformerBlock * basicTransformerBlock;

    struct tensor * project_out_weight;//[320,320,1,1]
    struct tensor * project_out_bias;//[320]
};

struct CrossAttention{
    int dim_head;//64
    int heads;//8
    int inner_dim;//8*64
    int scale;

    struct tensor * to_q;//[320,320]
    struct tensor * to_k;//[320,320]/[320, 768]
    struct tensor * to_v;//[320,320]/[320, 768]

    struct tensor * to_out_weight;//[320,320]
    struct tensor * to_out_bias;//[320]
};

struct FeedForward{
    struct tensor * linear_1_weight;//[2560,320]
    struct tensor * linear_1_bias;//[2560]

    struct tensor * linear_2_weight;//[320,1280]
    struct tensor * linear_2_bias;//[320]
};

struct input_blocks{
    struct tensor * PaddledConv2D;//[320, 4, 3, 3]
    struct tensor * PaddledConv2D_bias;//[320]

    struct ResBlock * ResBlock1;
    struct SpatialTransformer * SpatialTransformer1;
    struct ResBlock * ResBlock2;
    struct SpatialTransformer * SpatialTransformer2;
    struct tensor * downSample1_weight;//[320,320,3,3]
    struct tensor * downSample1_bias;//[320]
    struct ResBlock * ResBlock3;
    struct SpatialTransformer * SpatialTransformer3;
    struct ResBlock * ResBlock4;
    struct SpatialTransformer * SpatialTransformer4;
    struct tensor * downSample2_weight;//[640,640,3,3]
    struct tensor * downSample2_bias;//[640]
    struct ResBlock * ResBlock5;
    struct SpatialTransformer * SpatialTransformer5;
    struct ResBlock * ResBlock6;
    struct SpatialTransformer * SpatialTransformer6;
    struct tensor * downSample3_weight;//[1280,1280,3,3]
    struct tensor * downSample3_bias;//[1280]
    struct ResBlock * ResBlock7;
    struct ResBlock * ResBlock8;
};

struct middle_blocks {
    struct ResBlock * ResBlock1;
    struct SpatialTransformer * SpatialTransformer;
    struct ResBlock * ResBlock2;
};

struct output_blocks{
    struct ResBlock * ResBlock1;
    struct ResBlock * ResBlock2;
    struct ResBlock * ResBlock3;
    struct tensor * upSample1_weight;
    struct tensor * upSample1_bias;
    struct ResBlock * ResBlock4;
    struct SpatialTransformer * SpatialTransformer1;
    struct ResBlock * ResBlock5;
    struct SpatialTransformer * SpatialTransformer2;
    struct ResBlock * ResBlock6;
    struct SpatialTransformer * SpatialTransformer3;
    struct tensor * upSample2_weight;
    struct tensor * upSample2_bias;
    struct ResBlock * ResBlock7;
    struct SpatialTransformer * SpatialTransformer4;
    struct ResBlock * ResBlock8;
    struct SpatialTransformer * SpatialTransformer5;
    struct ResBlock * ResBlock9;
    struct SpatialTransformer * SpatialTransformer6;
    struct tensor * upSample3_weight;
    struct tensor * upSample3_bias;
    struct ResBlock * ResBlock10;
    struct SpatialTransformer * SpatialTransformer7;
    struct ResBlock * ResBlock11;
    struct SpatialTransformer * SpatialTransformer8;
    struct ResBlock * ResBlock12;
    struct SpatialTransformer * SpatialTransformer9;
};

struct unet_model {
    int model_channels;//320
    int num_res_blocks;//

    tensor* time_embed_0;//[320,1280]
    tensor* time_embed_0_bias;//[1280]
    tensor* time_embed_2;//[1280, 1280]
    tensor* time_embed_2_bias;//[1280]

    input_blocks* input_block;
    middle_blocks* middle_block;
    output_blocks* output_block;

    struct tensor* out_group_norm;
    struct tensor* out_group_norm_bias;

    struct tensor* out_corv;
    struct tensor* out_corv_bias;
};

struct tensor* conv(tensor *input, tensor *filter, int stride, int padding);
struct tensor * crossAttention_forward(struct tensor * input,struct tensor * context,struct CrossAttention * crossAttention);
struct tensor * GEGLU(struct tensor * input);
struct tensor * FeedForward_forward(struct tensor * input,struct FeedForward * feedForward);
struct tensor * BasicTransformerBlock_forward(struct tensor * input,struct tensor * context,struct BasicTransformerBlock * basicTransformerBlock);
struct unet_model* init_unet_model();
struct tensor * unet_forward(struct tensor * input,struct tensor * context,int timesteps,struct unet_model * unet);

#endif //UNTITLED_UNET_H
