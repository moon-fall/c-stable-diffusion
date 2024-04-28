//
// Created by 80324821 on 2023/4/21.
//

#include "unet.h"
#include "tensor.h"
#include "load.h"
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <cmath>
#include <cstring>

/*
对于多通道输入输出的卷积
输入X是3维的tensor 卷积K是4维的tensor
X->shape[0]=K->shape[1]
X的后两维和K的后两维 在X->shape[0]和K->shape[1]一一对应的情况下 做点积然后求和
输出Y是一个(K->shape[0] , (X->shape[1]+2*padding-K->shape[1])/stride+1 , (X->shape[2]+2*padding-K->shape[2])/stride+1)
*/
struct tensor* conv(tensor *input, tensor *filter, int stride, int padding) {
    //filter = transpose(filter,01,);

    // 计算输出tensor的高和宽
    int out_height = (input->shape[1] - filter->shape[2] + 2 * padding) / stride + 1;
    int out_width = (input->shape[2] - filter->shape[3] + 2 * padding) / stride + 1;

    // 创建输出tensor

    struct tensor* output = zeros_tensor(filter->shape[0],out_height,out_width);

    // 对每个输出通道执行卷积运算
    for (int oc = 0; oc < filter->shape[0]; ++oc) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                // 计算卷积结果
                float sum = 0.0f;
                for (int ic = 0; ic < filter->shape[1]; ++ic) {
                    for (int fh = 0; fh < filter->shape[2]; ++fh) {
                        for (int fw = 0; fw < filter->shape[3]; ++fw) {
                            int h = oh * stride + fh - padding;
                            int w = ow * stride + fw - padding;
                            if (h >= 0 && h < input->shape[1] && w >= 0 && w < input->shape[2]) {
                                float input_val = input->data[ic * input->shape[1] * input->shape[2] + h * input->shape[2] + w];
                                float filter_val = filter->data[oc * filter->shape[1] * filter->shape[2] * filter->shape[3]
                                                                + ic * filter->shape[2] * filter->shape[3] + fh * filter->shape[3] + fw];
                                sum += input_val * filter_val;
                            }
                        }
                    }
                }
                output->data[oc * out_height * out_width + oh * out_width + ow] = sum;
            }
        }
    }
    return output;
}

/*
基于lm2col和sgemm实现的卷积
卷积即输入矩阵中的子矩阵和卷积和做点积
我们可以将输入矩阵中的子矩阵展开成一维向量
然后点积就可以看做是计算矩阵相乘中的一个元素
比如输入为
 [[1,2,3,4],
  [5,6,7,8],
  [9,10,11,12],
  [13,14,15,16]]
卷积核为
 [[1,2,3],
 [4,5,6],
 [7,8,9]]

 输入形状为[b,c,h,w]

22                                                                                                                                                      `


 卷积核形状为[c_out,c_in,k,k]
 则如果stride为1 需要做(h-k)*(w-k)次子矩阵相乘
 这是单通道输入输出的需要做的点积次数
 同时[h-k,w-k]也是输出矩阵的大小我们记录做[out_h,out_w]
 我们执行Im2Col操作需要输入特征矩阵转化为 一个维度为卷积和大小的长度 另一个维度为输出矩阵大小的长度
 即[k*k,out_h*out_w]
 而卷积和的Im2Col操作则是展开成[1,k*k]
 两个矩阵相乘后得到[1,out_h*out_w]
 对于多输入的情况 最后也要加起来 实际上就是延长点积序列的长度
 比如输入通道c的情况下直接乘到k*k
 输入特征矩阵变为[in_c*k*k,out_h*out_w] 卷积和矩阵变为[1,in_c*k*k] 这样结果还是[1,out_h*out_w]
 再有就是输出通道数则加到卷积和的另一个维度上 变为 [in_c*k*k,out_h*out_w] * [out_c,in_c*k*k] = [out_c,out_h*out_w]

 */

struct tensor* im2col_input(struct tensor* input, struct tensor* kernel, int stride, int padding) {
    int in_c = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];
    int out_c = kernel->shape[0];
    int k_h = kernel->shape[2];
    int k_w = kernel->shape[3];
    int out_h = (in_h + 2 * padding - k_h) / stride + 1;
    int out_w = (in_w + 2 * padding - k_w) / stride + 1;

    int col_h = in_c * k_h * k_w;
    int col_w = out_h * out_w;
    int col_size = col_h * col_w;

    struct tensor* output = zeros_tensor(col_h,col_w);

    int col_idx = 0;
    for (int c = 0; c < in_c; c++) {
        for (int kh = 0; kh < k_h; kh++) {
            for (int kw = 0; kw < k_w; kw++) {
                for (int h = -padding; h < in_h + padding - k_h + 1; h += stride) {
                    for (int w = -padding; w < in_w + padding - k_w + 1; w += stride) {
                        int cur_h = h + kh;
                        int cur_w = w + kw;
                        if (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w) {
                            output->data[col_idx++] = input->data[((c * in_h) + cur_h) * in_w + cur_w];
                        }
                        else {
                            output->data[col_idx++] = 0;
                        }
                    }
                }
            }
        }
    }
    return output;
}

struct tensor* im2col_conv(struct tensor* input, struct tensor* filter, int stride, int padding){
    struct tensor* input_col = im2col_input(input,filter,stride,padding);
    int batch_size = input->shape[0];
    int in_c = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];
    int out_c = filter->shape[0];
    int k_h = filter->shape[2];
    int k_w = filter->shape[3];
    int out_h = (in_h + 2 * padding - k_h) / stride + 1;
    int out_w = (in_w + 2 * padding - k_w) / stride + 1;
    view(filter,out_c,in_c*k_h*k_w);
    struct tensor* output = mm2d(filter,input_col);
    view(filter,out_c,in_c,k_h,k_w);
    view(output,batch_size,out_c,out_h,out_w);
    return output;
}


struct tensor* conv_4d(struct tensor* input, struct tensor* filter, int stride, int padding) {
    // 计算输出张量形状
    int batch_size = input->shape[0];
    int input_channels = input->shape[1];
    int input_height = input->shape[2];
    int input_width = input->shape[3];
    int output_channels = filter->shape[0];
    int filter_height = filter->shape[2];
    int filter_width = filter->shape[3];
    int output_height = (input_height + 2 * padding - filter_height) / stride + 1;
    int output_width = (input_width + 2 * padding - filter_width) / stride + 1;

    // 分配输出张量所需的内存
    struct tensor* output = (struct tensor*)malloc(sizeof(struct tensor));
    output->dim = 4;
    output->shape = (int*)malloc(output->dim * sizeof(int));
    output->shape[0] = batch_size;
    output->shape[1] = output_channels;
    output->shape[2] = output_height;
    output->shape[3] = output_width;
    output->data = (float*)malloc(batch_size * output_channels * output_height * output_width * sizeof(float));

    // 对于每个数据点执行卷积运算
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < output_channels; oc++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    float value = 0.0;
                    for (int ic = 0; ic < input_channels; ic++) {
                        for (int fh = 0; fh < filter_height; fh++) {
                            for (int fw = 0; fw < filter_width; fw++) {
                                int ih = oh * stride + fh - padding;
                                int iw = ow * stride + fw - padding;
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    float input_value = input->data[b * input_channels * input_height * input_width
                                                                    + ic * input_height * input_width
                                                                    + ih * input_width
                                                                    + iw];
                                    float filter_value = filter->data[oc * input_channels * filter_height * filter_width
                                                                      + ic * filter_height * filter_width
                                                                      + fh * filter_width
                                                                      + fw];
                                    value += input_value * filter_value;
                                }
                            }
                        }
                    }
                    output->data[b * output_channels * output_height * output_width
                                 + oc * output_height * output_width
                                 + oh * output_width
                                 + ow] = value;
                }
            }
        }
    }

    // 返回输出张量
    return output;
}

struct tensor* MaxPool2d(struct tensor* X,int* pool_size){
    struct tensor* Y = zeros_tensor(X->shape[0]-pool_size[0]+1,X->shape[1]-pool_size[1]+1);
    for(int i=0;i<Y->shape[0];i++){
        for(int j=0;j<Y->shape[1];j++){
            float max=FLT_MIN;
            for(int k=i;k<i+pool_size[0];k++){
                for(int l=j;l<j+pool_size[1];l++){
                    printf("%d %d %d %d %d\n",i,j,k,l,k*X->shape[1]+l);
                    float tmp = X->data[k*X->shape[1]+l];
                    max = max<tmp?tmp:max;
                }
            }
            Y->data[i*Y->shape[1]+j]=max;
        }
    }
    return Y;
}

struct tensor* timestep_embedding(int timesteps,int dim,int max_period){
    struct tensor* output =  zeros_tensor(1,dim);
    for (int i = 0; i < dim; i++) {
        float exponent = 2 * timesteps / dim;
        float angle = i / pow(max_period, exponent);
        if(i%2==0){
            output->data[i]=sin(angle);
        }else{
            output->data[i]=cos(angle);
        }
    }
    return output;
}

struct tensor * nearest_interpolate(struct tensor * input,int output_h,int output_w){
    int in_h = input->shape[0];
    int in_w = input->shape[1];
    struct tensor * output = zeros_tensor(output_h,output_w);

    int out_h = output->shape[0];
    int out_w = output->shape[1];
    float scale_h = (float)in_h / out_h;
    float scale_w = (float)in_w / out_w;
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            int in_i = (int)(i * scale_h);
            int in_j = (int)(j * scale_w);
            output->data[i * out_w + j] = input->data[in_i * in_w + in_j];
        }
    }
    return output;
}

struct tensor * nearest_interpolate_2d(struct tensor * input,int out_h,int out_w){
    int in_len = input->shape[2]*input->shape[3];
    int out_len = out_h*out_w;
    struct tensor * output = zeros_tensor(input->shape[0],input->shape[1],out_h,out_w);
    int in_h = input->shape[2];
    int in_w = input->shape[3];
    float scale_h = (float)in_h / out_h;
    float scale_w = (float)in_w / out_w;
    for (int step = 0; step < input->shape[0] * input->shape[1]; step++) {
        int in_start_index = step * in_len;
        int out_start_index = step * out_len;
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                int in_i = (int) (i * scale_h);
                int in_j = (int) (j * scale_w);
                output->data[out_start_index + i * out_w + j] = input->data[in_start_index + in_i * in_w + in_j];
            }
        }
    }
    return output;
}


struct tensor * nearest_interpolate_3d(struct tensor * input,int out_d,int out_h,int out_w){
    int in_len = input->shape[2]*input->shape[3]*input->shape[4];
    int out_len = out_d*out_h*out_w;

    struct tensor * output = zeros_tensor(input->shape[0],input->shape[1],out_d,out_h,out_w);

    int in_d = input->shape[2];
    int in_h = input->shape[3];
    int in_w = input->shape[4];
    float scale_d = (float)in_d / out_d;
    float scale_h = (float)in_h / out_h;
    float scale_w = (float)in_w / out_w;
    for(int step=0;step<input->shape[0]*input->shape[1];step++){
        int in_start_index = step*in_len;
        int out_start_index = step*out_len;
        for (int i = 0; i < out_d; i++) {
            for (int j = 0; j < out_h; j++) {
                for (int k = 0; k < out_w; k++) {
                    int in_i = (int) (i * scale_d);
                    int in_j = (int) (j * scale_h);
                    int in_k = (int) (k * scale_w);
                    output->data[out_start_index + i * out_h * out_w + j*out_w + k] = input->data[in_start_index + in_i * in_h * in_w + in_j * in_w + in_k];
                }
            }
        }
    }
    return output;
}

struct tensor * upSample(struct tensor * input,struct tensor * filter){
    struct tensor * output = nearest_interpolate_2d(input,input->shape[2]*2,input->shape[3]*2);
    output = im2col_conv(output,filter,1,1);
    return output;
}

struct tensor * downSample(struct tensor * input,struct tensor * filter){
    struct tensor * output = im2col_conv(input,filter,2,1);
    return output;
}

struct tensor * in_layer_forward(struct tensor * input,struct in_layers * in_layer){
    struct tensor * output = group_norm_4d(32,input);
    silu_tensor(output);
    //output = conv_4d(output,in_layer->conv_weight,1,1);
    output = im2col_conv(output,in_layer->conv_weight,1,1);
    return output;
}

//input:[1,320,64,64]
struct tensor * out_layer_forward(struct tensor * input,struct out_layers * out_layer){
    //view(input,1,input->shape[0],input->shape[1],input->shape[2]);
    struct tensor * output = group_norm_4d(32,input);
    silu_tensor(output);
//    output = conv_4d(output,out_layer->conv_weight,1,1);
    output = im2col_conv(output,out_layer->conv_weight,1,1);
    view(out_layer->conv_bias,1,out_layer->conv_weight->shape[0],1,1);
    auto_broadcast_add(output,out_layer->conv_bias);
    return output;
}

//input:[1,1028]   return:[1,320]
struct tensor * emb_layers_forward(struct tensor * input,struct emb_layers * emb_layer){
    silu_tensor(input);
    struct tensor * output = linear(input,emb_layer->linear_weight);
    output= add(output,emb_layer->linear_bias, true);
    return output;
}

struct tensor * res_block_forward(struct tensor * input,struct tensor * emb,struct ResBlock * resBlock){
    struct tensor * h = in_layer_forward(input,resBlock->in_layer);
    struct tensor * emb_out = emb_layers_forward(emb,resBlock->emb_layer);
    view(emb_out,emb_out->shape[0],emb_out->shape[1],1,1);
    auto_broadcast_add(h,emb_out);
    h = out_layer_forward(h,resBlock->out_layer);
    if(resBlock->skip_connection_weight!=NULL){
        input = im2col_conv(input,resBlock->skip_connection_weight,1,0);
    }
    add(input,h, true);
    return input;
}

//input [1,320,64,64]  context [1,77,768]
struct tensor * SpatialTransformer_forward(struct tensor * input,struct tensor * context,struct SpatialTransformer * spatialTransformer){
    int h = input->shape[2];
    int w = input->shape[3];
    int c = input->shape[1];
    struct tensor * output = group_norm_4d(32,input);
//    output = conv_4d(output,spatialTransformer->project_in_weight,1,0);
    output = im2col_conv(output,spatialTransformer->project_in_weight,1,0);
    view(output,output->shape[0],output->shape[1],output->shape[2]*output->shape[3]);
    output = transpose(output,1,2);
    output = BasicTransformerBlock_forward(output,context,spatialTransformer->basicTransformerBlock);//[1,4096,320]
    output = transpose(output,1,2);
    view(output,1,c,h,w);
    //output = conv_4d(output,spatialTransformer->project_out_weight,1,0);
    output = im2col_conv(output,spatialTransformer->project_out_weight,1,0);
    add(output,input, true);
    return output;
}

struct tensor * BasicTransformerBlock_forward(struct tensor * input,struct tensor * context,struct BasicTransformerBlock * basicTransformerBlock){
    tensor * output = add(input,crossAttention_forward(layer_norm(input),NULL,basicTransformerBlock->attn1));
    output = add(output,crossAttention_forward(layer_norm(output),context,basicTransformerBlock->attn2));
    output = FeedForward_forward(layer_norm(output),basicTransformerBlock->ff);
    return output;
}

//intput:[1,4096,320]
struct tensor * crossAttention_forward(struct tensor * input,struct tensor * context,struct CrossAttention * crossAttention){
    int dim_head = crossAttention->dim_head;
    int head = crossAttention->heads;
    int inner_dim = dim_head * head;

    tensor * q = linear(input,crossAttention->to_q);
    context=context == NULL?input:context;
    int input_len=input->shape[1];
    int context_len=context->shape[1];
    tensor * k = linear(context,crossAttention->to_k);
    tensor * v = linear(context,crossAttention->to_v);

    view(q,1,input_len,head,dim_head);
    transpose(q,1,2);
    view(q,head,input_len,dim_head);

    view(k,1,context_len,head,dim_head);
    transpose(k,1,2);
    view(k,head,context_len,dim_head);

    view(v,1,context_len,head,dim_head);
    transpose(v,1,2);
    view(v,head,context_len,dim_head);

    tensor* scores = tensor_scaled_division(mm3d(q,transpose(k,1,2)),sqrt(dim_head));

    softmax_last_dim(scores);
    tensor* output = mm3d(scores,v);
    view(output,1,head,input_len,dim_head);
    output = view(transpose(output,1,2),1,input_len,inner_dim);
    output = linear(output,crossAttention->to_out_weight);
    return output;
}

//input:[1,4096,320]
struct tensor * FeedForward_forward(struct tensor * input,struct FeedForward * feedForward){
    struct tensor * output = linear(input,feedForward->linear_1_weight);
    view(feedForward->linear_1_bias,1,1,feedForward->linear_1_bias->shape[0]);
    auto_broadcast_add(output,feedForward->linear_1_bias);
    view(feedForward->linear_1_bias,feedForward->linear_1_bias->shape[2]);
    output = GEGLU(output);//[4096,1280]
    output = linear(output,feedForward->linear_2_weight);
    view(feedForward->linear_2_bias,1,1,feedForward->linear_2_bias->shape[0]);
    auto_broadcast_add(output,feedForward->linear_2_bias);
    view(feedForward->linear_1_bias,feedForward->linear_2_bias->shape[2]);
    return output;
}

struct tensor * GEGLU(struct tensor * input){
    int * output_shape = (int*) malloc(input->dim);
    memcpy(output_shape, input->shape, input->dim * sizeof(int));
    output_shape[input->dim-1]/=2;
    struct tensor * output = zeros_tensor(input->dim,output_shape);
    for(int i=0;i< get_tensor_size(input)/input->shape[input->dim-1];i++){
        int input_index = i * input->shape[input->dim-1];
        int output_index = i * output->shape[input->dim-1];
        for(int j=0;j<output->shape[input->dim-1];j++){
            output->data[output_index+j]=input->data[input_index+j]*gelu(input->data[input_index+j+output->shape[input->dim-1]]);
        }
    }
    return output;
}

struct tensor * resnet_and_spatial_forward(struct tensor * input,struct tensor * emb,struct tensor * context,struct ResBlock * resBlock,struct SpatialTransformer * spatialTransformer){
    clock_t t1 = clock();
    struct tensor * output = res_block_forward(input,emb,resBlock);
    clock_t t2 = clock();
    output = SpatialTransformer_forward(output,context,spatialTransformer);
    clock_t t3 = clock();
    printf("SpatialTransformer_forward cost %lf ms\n",(double)(t3-t2)/1000);
    return output;
}

struct tensor * input_blocks_forward(struct tensor * input,struct tensor * emb,struct tensor * context,struct input_blocks * input_block,std::vector<tensor*>& vec){
    clock_t t1 = clock();
    //struct tensor * hidden1 = conv_4d(input,input_block->PaddledConv2D,1,1);
    printf("input_block->PaddledConv2D");
    shape_print(input_block->PaddledConv2D);
    printf("input");
    shape_print(input);

    struct tensor * hidden1 = im2col_conv(input,input_block->PaddledConv2D,1,1);
    printf("hiddin1\n");
    shape_print(hidden1);

    clock_t t2 = clock();
    printf("PaddledConv2D cost %lf ms\n",(double)(t2-t1)/1000);
    vec.push_back(hidden1);
    struct tensor * hidden2 = resnet_and_spatial_forward(hidden1,emb,context,input_block->ResBlock1,input_block->SpatialTransformer1);
    clock_t t3 = clock();
    printf("resnet_and_spatial_forward cost %lf ms\n",(double)(t3-t2)/1000);
    vec.push_back(hidden2);
    struct tensor * hidden3 = resnet_and_spatial_forward(hidden2,emb,context,input_block->ResBlock2,input_block->SpatialTransformer2);
    vec.push_back(hidden3);
    printf("hidden3:");
    shape_print(hidden3);
    printf("input_block->downSample1_weight:");
    shape_print(input_block->downSample1_weight);
    struct tensor * hidden4 = downSample(hidden3,input_block->downSample1_weight);
    printf("hidden4:");
    shape_print(hidden4);
    vec.push_back(hidden4);
    struct tensor * hidden5 = resnet_and_spatial_forward(hidden4,emb,context,input_block->ResBlock3,input_block->SpatialTransformer3);
    printf("hidden5:");
    shape_print(hidden5);
    vec.push_back(hidden5);
    struct tensor * hidden6 = resnet_and_spatial_forward(hidden5,emb,context,input_block->ResBlock4,input_block->SpatialTransformer4);
    vec.push_back(hidden6);
    printf("hidden6:");
    shape_print(hidden6);
    struct tensor * hidden7 = downSample(hidden6,input_block->downSample2_weight);
    vec.push_back(hidden7);
    printf("hidden7:");
    shape_print(hidden7);
    struct tensor * hidden8 = resnet_and_spatial_forward(hidden7,emb,context,input_block->ResBlock5,input_block->SpatialTransformer5);
    vec.push_back(hidden8);
    printf("hidden8:");
    shape_print(hidden8);
    struct tensor * hidden9 = resnet_and_spatial_forward(hidden8,emb,context,input_block->ResBlock6,input_block->SpatialTransformer6);
    vec.push_back(hidden9);
    printf("hidden9:");
    shape_print(hidden9);
    struct tensor * hidden10 = downSample(hidden9,input_block->downSample3_weight);
    vec.push_back(hidden10);
    printf("hidden10:");
    shape_print(hidden10);
    struct tensor * hidden11 = res_block_forward(hidden10,emb,input_block->ResBlock7);
    vec.push_back(hidden11);
    printf("hidden11:");
    shape_print(hidden11);
    struct tensor * hidden12 = res_block_forward(hidden11,emb,input_block->ResBlock8);
    vec.push_back(hidden12);
    printf("hidden12:");
    shape_print(hidden12);
    return hidden12;
}

//input:[1,1280,8,8]
struct tensor * output_blocks_forward(struct tensor * input,struct tensor * emb,struct tensor * context,struct output_blocks * output_block,std::vector<tensor*>& vec){
    struct tensor * hidden12 = vec.back();
    struct tensor * output = cat(input,hidden12,1);
    output = res_block_forward(output,emb,output_block->ResBlock1);
    vec.pop_back();
    struct tensor * hidden11 = vec.back();
    output = cat(output,hidden11,1);
    output = res_block_forward(output,emb,output_block->ResBlock2);
    vec.pop_back();
    struct tensor * hidden10 = vec.back();
    output = cat(output,hidden10,1);
    output = res_block_forward(output,emb,output_block->ResBlock3);
    output = upSample(output,output_block->upSample1_weight);
    vec.pop_back();
    struct tensor * hidden9 = vec.back();
    output = cat(output,hidden9,1);
    output = res_block_forward(output,emb,output_block->ResBlock4);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer1);
    vec.pop_back();
    struct tensor * hidden8 = vec.back();
    output = cat(output,hidden8,1);
    output = res_block_forward(output,emb,output_block->ResBlock5);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer2);
    vec.pop_back();
    struct tensor * hidden7 = vec.back();
    printf("hidden7:");
    shape_print(hidden7);
    output = cat(output,hidden7,1);
    printf("output+hidden7:");
    shape_print(output);
    output = res_block_forward(output,emb,output_block->ResBlock6);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer3);
    output = upSample(output,output_block->upSample2_weight);
    vec.pop_back();
    struct tensor * hidden6 = vec.back();
    output = cat(output,hidden6,1);
    output = res_block_forward(output,emb,output_block->ResBlock7);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer4);
    vec.pop_back();
    struct tensor * hidden5 = vec.back();
    output = cat(output,hidden5,1);
    output = res_block_forward(output,emb,output_block->ResBlock8);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer5);
    vec.pop_back();
    struct tensor * hidden4 = vec.back();
    output = cat(output,hidden4,1);
    output = res_block_forward(output,emb,output_block->ResBlock9);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer6);
    output = upSample(output,output_block->upSample3_weight);
    vec.pop_back();
    struct tensor * hidden3 = vec.back();
    output = cat(output,hidden3,1);
    output = res_block_forward(output,emb,output_block->ResBlock10);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer7);
    vec.pop_back();
    struct tensor * hidden2 = vec.back();
    output = cat(output,hidden2,1);
    output = res_block_forward(output,emb,output_block->ResBlock11);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer8);
    vec.pop_back();
    struct tensor * hidden1 = vec.back();
    output = cat(output,hidden1,1);
    output = res_block_forward(output,emb,output_block->ResBlock12);
    output = SpatialTransformer_forward(output,context,output_block->SpatialTransformer9);
    return output;
}

struct tensor * middle_block_forward(struct tensor * input,struct tensor * emb,struct tensor * context,struct middle_blocks* middle_block){
    struct tensor * output = res_block_forward(input,emb,middle_block->ResBlock1);
    output = SpatialTransformer_forward(output,context,middle_block->SpatialTransformer);
    output = res_block_forward(input,emb,middle_block->ResBlock2);
    return output;
}

struct tensor * unet_forward(struct tensor * input,struct tensor * context,int timesteps,struct unet_model * unet){
    struct tensor* t_emb = timestep_embedding(timesteps,unet->model_channels,10000);
    t_emb = mm2d(t_emb,unet->time_embed_0);
    silu_tensor(t_emb);
    t_emb = mm2d(t_emb,unet->time_embed_2);
    std::vector<struct tensor*> vec;
    struct tensor * output = input_blocks_forward(input,t_emb,context,unet->input_block,vec);
    output = middle_block_forward(output,t_emb,context,unet->middle_block);
    output = output_blocks_forward(output,t_emb,context,unet->output_block,vec);
    output = im2col_conv(output,unet->out_corv,1,1);
    return output;
}

struct in_layers* init_in_layers(int channel,int out_channel){
    struct in_layers* in_layer = (struct in_layers*)malloc(sizeof(in_layers));
    in_layer->group_norm_wight = zeros_tensor(channel);
    in_layer->group_norm_bias = zeros_tensor(channel);
    in_layer->conv_weight = zeros_tensor(out_channel,channel,3,3);
    in_layer->conv_bias = zeros_tensor(out_channel);
    return in_layer;
}

struct emb_layers* init_emb_layers(int channel){
    struct emb_layers* emb_layer = (struct emb_layers*)malloc(sizeof(emb_layers));
    emb_layer->linear_weight = zeros_tensor(channel,1280);
    emb_layer->linear_bias = zeros_tensor(channel);
    return emb_layer;
}

struct out_layers* init_out_layers(int channel){
    struct out_layers* out_layer = (struct out_layers*)malloc(sizeof(out_layers));
    out_layer->group_norm_wight = zeros_tensor(channel);
    out_layer->group_norm_bias = zeros_tensor(channel);
    out_layer->conv_weight = zeros_tensor(channel,channel,3,3);
    out_layer->conv_bias = zeros_tensor(channel);
    return out_layer;
}

struct ResBlock* init_ResBlock(int in_channel,int out_channel){
    struct ResBlock* resBlock = (struct ResBlock*)malloc(sizeof(resBlock));
    resBlock->in_layer = init_in_layers(in_channel,out_channel);
    resBlock->emb_layer = init_emb_layers(out_channel);
    resBlock->out_layer = init_out_layers(out_channel);
    resBlock->skip_connection_weight=in_channel==out_channel?NULL:zeros_tensor(out_channel,in_channel,1,1);
    resBlock->skip_connection_bias=in_channel==out_channel?NULL: zeros_tensor(out_channel);
    return resBlock;
};

struct FeedForward* init_FeedForward(int dim){
    struct FeedForward* feedForward = (struct FeedForward*)malloc(sizeof(FeedForward));
    int inner_dim = dim*4;
    feedForward->linear_1_weight= zeros_tensor(inner_dim*2,dim);
    feedForward->linear_1_bias = zeros_tensor(inner_dim*2);
    feedForward->linear_2_weight= zeros_tensor(dim,inner_dim);
    feedForward->linear_2_bias = zeros_tensor(dim);
    return feedForward;
}

struct CrossAttention* init_CrossAttention(int query_dim,int context_dim,int heads,int dim_head){
    struct CrossAttention* crossAttention = (struct CrossAttention*)malloc(sizeof(CrossAttention));
    int inner_dim = dim_head * heads;
    crossAttention->dim_head=dim_head;
    crossAttention->heads=heads;
    crossAttention->to_q = zeros_tensor(inner_dim,query_dim);
    crossAttention->to_k = zeros_tensor(inner_dim,context_dim);
    crossAttention->to_v = zeros_tensor(inner_dim,context_dim);
    crossAttention->to_out_weight = zeros_tensor(query_dim,inner_dim);
    crossAttention->to_out_bias = zeros_tensor(query_dim);
    return crossAttention;
}

struct BasicTransformerBlock* init_BasicTransformerBlock(int dim,int n_heads,int d_head,int context_dim){
    struct BasicTransformerBlock* basicTransformerBlock = (struct BasicTransformerBlock*)malloc(sizeof(BasicTransformerBlock));
    basicTransformerBlock->attn1 = init_CrossAttention(dim,dim,n_heads,d_head);
    basicTransformerBlock->ff = init_FeedForward(dim);
    basicTransformerBlock->attn2 = init_CrossAttention(dim,context_dim,n_heads,d_head);
    basicTransformerBlock->layer_norm_wight_1= zeros_tensor(dim);
    basicTransformerBlock->layer_norm_wight_2= zeros_tensor(dim);
    basicTransformerBlock->layer_norm_wight_3= zeros_tensor(dim);
    basicTransformerBlock->layer_norm_bias_1= zeros_tensor(dim);
    basicTransformerBlock->layer_norm_bias_2= zeros_tensor(dim);
    basicTransformerBlock->layer_norm_bias_3= zeros_tensor(dim);
    return basicTransformerBlock;
}

struct SpatialTransformer* init_SpatialTransformer(int in_channels,int n_heads,int d_head,int context_dim){
    struct SpatialTransformer* spatialTransformer = (struct SpatialTransformer*)malloc(sizeof(SpatialTransformer));
    int inner_dim = n_heads * d_head;
    spatialTransformer->group_norm_weight = zeros_tensor(in_channels);
    spatialTransformer->group_norm_bias = zeros_tensor(in_channels);
    spatialTransformer->project_in_weight= zeros_tensor(in_channels,inner_dim,1,1);
    spatialTransformer->project_in_bias = zeros_tensor(in_channels);
    spatialTransformer->basicTransformerBlock = init_BasicTransformerBlock(inner_dim,n_heads,d_head,context_dim);
    spatialTransformer->project_out_weight = zeros_tensor(in_channels,inner_dim,1,1);
    spatialTransformer->project_out_bias = zeros_tensor(in_channels);
    return spatialTransformer;
}

struct input_blocks* init_input_blocks(){
    int model_channels=320;
    struct input_blocks* input_block = (struct input_blocks*)malloc(sizeof(input_blocks));
    input_block->PaddledConv2D = zeros_tensor(320,4,3,3);
    input_block->PaddledConv2D_bias = zeros_tensor(320);
    input_block->ResBlock1 = init_ResBlock(320,320);
    input_block->SpatialTransformer1 = init_SpatialTransformer(320,8,40,768);
    input_block->ResBlock2 = init_ResBlock(320,320);
    input_block->SpatialTransformer2 = init_SpatialTransformer(320,8,40,768);
    input_block->downSample1_weight = zeros_tensor(320,320,3,3);
    input_block->downSample1_bias = zeros_tensor(320);
    input_block->ResBlock3 = init_ResBlock(320,640);
    input_block->SpatialTransformer3 = init_SpatialTransformer(640,8,80,768);
    input_block->ResBlock4 = init_ResBlock(640,640);
    input_block->SpatialTransformer4 = init_SpatialTransformer(640,8,80,768);
    input_block->downSample2_weight = zeros_tensor(640,640,3,3);
    input_block->downSample2_bias = zeros_tensor(640);
    input_block->ResBlock5 = init_ResBlock(640,1280);
    input_block->SpatialTransformer5 = init_SpatialTransformer(1280,8,160,768);
    input_block->ResBlock6 = init_ResBlock(1280,1280);
    input_block->SpatialTransformer6 = init_SpatialTransformer(1280,8,160,768);
    input_block->downSample3_weight = zeros_tensor(1280,1280,3,3);
    input_block->downSample3_bias = zeros_tensor(1280);
    input_block->ResBlock7 = init_ResBlock(1280,1280);
    input_block->ResBlock8 = init_ResBlock(1280,1280);
    return input_block;
}

struct middle_blocks* init_middle_blocks(){
    struct middle_blocks* middle_block = (struct middle_blocks*)malloc(sizeof(middle_blocks));
    middle_block->ResBlock1= init_ResBlock(1280,1280);
    middle_block->SpatialTransformer = init_SpatialTransformer(1280,8,160,768);
    middle_block->ResBlock2= init_ResBlock(1280,1280);
    return middle_block;
}

struct output_blocks* init_output_blocks(){
    struct output_blocks* output_block = (struct output_blocks*)malloc(sizeof(output_blocks));
    output_block->ResBlock1= init_ResBlock(2560,1280);
    output_block->ResBlock2= init_ResBlock(2560,1280);
    output_block->ResBlock3= init_ResBlock(2560,1280);
    output_block->upSample1_weight = zeros_tensor(1280,1280,3,3);
    output_block->upSample1_bias = zeros_tensor(1280);
    output_block->ResBlock4= init_ResBlock(2560,1280);
    output_block->SpatialTransformer1 = init_SpatialTransformer(1280,8,160,768);
    output_block->ResBlock5= init_ResBlock(2560,1280);
    output_block->SpatialTransformer2 = init_SpatialTransformer(1280,8,160,768);
    output_block->ResBlock6= init_ResBlock(1920,1280);
    output_block->SpatialTransformer3 = init_SpatialTransformer(1280,8,160,768);
    output_block->upSample2_weight = zeros_tensor(1280,1280,3,3);
    output_block->upSample2_bias = zeros_tensor(1280);
    output_block->ResBlock7= init_ResBlock(1920,640);
    output_block->SpatialTransformer4 = init_SpatialTransformer(640,8,80,768);
    output_block->ResBlock8= init_ResBlock(1280,640);
    output_block->SpatialTransformer5 = init_SpatialTransformer(640,8,80,768);
    output_block->ResBlock9= init_ResBlock(960,640);
    output_block->SpatialTransformer6 = init_SpatialTransformer(640,8,80,768);
    output_block->upSample3_weight = zeros_tensor(640,640,3,3);
    output_block->upSample3_bias = zeros_tensor(640);
    output_block->ResBlock10= init_ResBlock(960,320);
    output_block->SpatialTransformer7 = init_SpatialTransformer(320,8,40,768);
    output_block->ResBlock11= init_ResBlock(640,320);
    output_block->SpatialTransformer8 = init_SpatialTransformer(320,8,40,768);
    output_block->ResBlock12= init_ResBlock(640,320);
    output_block->SpatialTransformer9 = init_SpatialTransformer(320,8,40,768);
    return output_block;
}

struct unet_model* init_unet_model(){
    struct unet_model* unet = (struct unet_model*)malloc(sizeof(unet_model));
    unet->time_embed_0 = zeros_tensor(1280,320);
    unet->time_embed_0_bias = zeros_tensor(1280);
    unet->time_embed_2 = zeros_tensor(1280,1280);
    unet->time_embed_2_bias = zeros_tensor(1280);
    unet->input_block = init_input_blocks();
    unet->middle_block = init_middle_blocks();
    unet->output_block = init_output_blocks();
    unet->out_group_norm = zeros_tensor(320);
    unet->out_group_norm_bias = zeros_tensor(320);
    unet->out_corv = zeros_tensor(4,320,3,3);
    unet->out_corv_bias = zeros_tensor(4);
    return unet;
}


void test_corr2d(){
    float x_arr[18] = {0.0, 1.0, 2.0, 3.0, 4.0,5.0, 6.0, 7.0, 8.0
            ,1.0, 2.0, 3.0, 4.0,5.0, 6.0, 7.0, 8.0,9.0};
    struct tensor * X = tensor_from_array(x_arr,2,3,3);
    tensor_print(X);
    float k_arr[54] = {0.0,1.0,2.0,3.0,1.0,2.0,3.0,4.0,
                       1.0,2.0,3.0,4.0,2.0,3.0,4.0,5.0,
                       2.0,3.0,4.0,5.0,3.0,4.0,5.0,6.0};
    struct tensor * K = tensor_from_array(k_arr,3,2,2,2);
    tensor_print(K);
    struct tensor * Y = conv(X,K,1,0);
    tensor_print(Y);
}

void im2col_conv(){
    struct tensor* X = ones_tensor(1,320,64,64);
    struct tensor* K = ones_tensor(320,320,3,3);

    clock_t t1 = clock();
    struct tensor* output = im2col_conv(X,K,1,0);
    clock_t t2 = clock();
    printf("im2col_conv cost %lf ms\n",(double)(t2-t1)/1000);
    struct tensor* output2 = conv_4d(X,K,1,0);
    clock_t t3 = clock();
    printf("conv_4d cost %lf ms\n",(double)(t3-t2)/1000);
    shape_print(output);
    shape_print(output2);
}

void test_MaxPool2d(){
    struct tensor* X = seq_tensor(3,3);
    tensor_print(X);
    int pool_size[2] = {2,2};
    struct tensor* Y = MaxPool2d(X,pool_size);
    tensor_print(Y);
}

void test_nearest_interpolate(){
    struct tensor* X = seq_tensor(1,2,2,3,3);
    tensor_print(X);
    struct tensor* Y = nearest_interpolate_3d(X,2,6,6);
    tensor_print(Y);
}

void test_GEGLU(){
    struct tensor * X =  seq_tensor(2,3,4);
    X = GEGLU(X);
    tensor_print(X);
}

void test_SpatialTransformer_forward(){
    struct tensor * input = rand_tensor(1,320,64,64);
    struct tensor * context = rand_tensor(1,77,768);
    SpatialTransformer* spatialTransformer = init_SpatialTransformer(320,8,40,768);
    struct tensor * output = SpatialTransformer_forward(input,context,spatialTransformer);
    shape_print(output);
}

void test_crossAttention_forward(){
    struct tensor * input = rand_tensor(1,4096,320);
    struct tensor * context = rand_tensor(1,77,768);
    struct CrossAttention* selfAttention = init_CrossAttention(320,320,8,40);
    struct tensor * output = crossAttention_forward(input,input,selfAttention);
    struct CrossAttention* crossAttention = init_CrossAttention(320,768,8,40);
    output = crossAttention_forward(output,context,crossAttention);
    shape_print(output);
}

void test_BasicTransformerBlock_forward(){
    struct tensor * input = rand_tensor(1,4096,320);
    struct tensor * context = rand_tensor(1,77,768);
    struct BasicTransformerBlock * basicTransformerBlock = init_BasicTransformerBlock(320,8,40,768);
    struct tensor * output = BasicTransformerBlock_forward(input,context,basicTransformerBlock);
    shape_print(output);
}

void test_res_block_forward(){
    struct ResBlock* resBlock = init_ResBlock(320,640);
    struct tensor* input = rand_tensor(1,320,32,32);
    struct tensor* emb = rand_tensor(1,1280);
    struct tensor* output =  res_block_forward(input,emb,resBlock);
    shape_print(output);
}

void test_input_blocks(){
    struct input_blocks* input_block =  init_input_blocks();
    struct tensor* input = rand_tensor(1,4,64,64);
    struct tensor* emb = rand_tensor(1,1280);
    struct tensor * context = rand_tensor(1,77,768);
    std::vector<struct tensor*> vec;
    struct tensor * output = input_blocks_forward(input,emb,context,input_block,vec);
    shape_print(output);
}

void test_middle_blocks(){
    struct middle_blocks* middle_block =  init_middle_blocks();
    struct tensor* input = rand_tensor(1,1280,8,8);
    struct tensor* emb = rand_tensor(1,1280);
    struct tensor * context = rand_tensor(1,77,768);
    struct tensor * output = middle_block_forward(input,emb,context,middle_block);
    shape_print(output);
}

void test_output_blocks(){
    struct output_blocks* output_block =  init_output_blocks();
    struct tensor* input = rand_tensor(1,1280,8,8);
    struct tensor* emb = rand_tensor(1,1280);
    struct tensor * context = rand_tensor(1,77,768);
    std::vector<struct tensor*> vec;
    vec.push_back(zeros_tensor(1,320,64,64));
    vec.push_back(zeros_tensor(1,320,64,64));
    vec.push_back(zeros_tensor(1,320,64,64));
    vec.push_back(zeros_tensor(1,320,32,32));

    vec.push_back(zeros_tensor(1,640,32,32));
    vec.push_back(zeros_tensor(1,640,32,32));

    vec.push_back(zeros_tensor(1,640,16,16));

    vec.push_back(zeros_tensor(1,1280,16,16));
    vec.push_back(zeros_tensor(1,1280,16,16));

    vec.push_back(zeros_tensor(1,1280,8,8));
    vec.push_back(zeros_tensor(1,1280,8,8));
    vec.push_back(zeros_tensor(1,1280,8,8));

    struct tensor * output = output_blocks_forward(input,emb,context,output_block,vec);
    shape_print(output);
}

void test_unet_forward(){
    struct unet_model* unet = init_unet_model();
    struct tensor* input = rand_tensor(1,4,64,64);
    struct tensor* emb = rand_tensor(1,1280);
    struct tensor * context = rand_tensor(1,77,768);
    clock_t t1 = clock();
    struct tensor* output = unet_forward(input,context,1,unet);
    clock_t t2 = clock();
    printf("unet forward cost %lf ms\n",(double)(t2-t1)/1000);
}


//int main() {
//    //test_input_blocks();
//    test_unet_forward();
//};
