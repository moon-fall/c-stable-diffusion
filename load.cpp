#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>

#include "unet.h"
#include "load.h"

void load_FeedForward(struct FeedForward * feedForward,std::ifstream& file){
    load_to_mem(feedForward->linear_1_weight,file);
    load_to_mem(feedForward->linear_1_bias,file);
    load_to_mem(feedForward->linear_2_weight,file);
    load_to_mem(feedForward->linear_2_bias,file);
}

void load_CrossAttention(struct CrossAttention * crossAttention,std::ifstream& file){
    load_to_mem(crossAttention->to_q,file);
    load_to_mem(crossAttention->to_k,file);
    load_to_mem(crossAttention->to_v,file);

    load_to_mem(crossAttention->to_out_weight,file);
    load_to_mem(crossAttention->to_out_bias,file);
}

void load_BasicTransformerBlock(struct BasicTransformerBlock * basicTransformerBlock,std::ifstream& file){
    load_CrossAttention(basicTransformerBlock->attn1,file);
    load_FeedForward(basicTransformerBlock->ff,file);
    load_CrossAttention(basicTransformerBlock->attn2,file);
    load_to_mem(basicTransformerBlock->layer_norm_wight_1,file);
    load_to_mem(basicTransformerBlock->layer_norm_bias_1,file);
    load_to_mem(basicTransformerBlock->layer_norm_wight_2,file);
    load_to_mem(basicTransformerBlock->layer_norm_bias_2,file);
    load_to_mem(basicTransformerBlock->layer_norm_wight_3,file);
    load_to_mem(basicTransformerBlock->layer_norm_bias_3,file);
}

void load_SpatialTransformer(struct SpatialTransformer * spatialTransformer,std::ifstream& file){
    load_to_mem(spatialTransformer->group_norm_weight,file);
    load_to_mem(spatialTransformer->group_norm_bias,file);
    load_to_mem(spatialTransformer->project_in_weight,file);
    load_to_mem(spatialTransformer->project_in_bias,file);
    load_BasicTransformerBlock(spatialTransformer->basicTransformerBlock,file);
    load_to_mem(spatialTransformer->project_out_weight,file);
    load_to_mem(spatialTransformer->project_out_bias,file);
}

void load_out_layers(struct out_layers * out_layer,std::ifstream& file){
    load_to_mem(out_layer->group_norm_wight,file);
    load_to_mem(out_layer->group_norm_bias,file);
    load_to_mem(out_layer->conv_weight,file);
    load_to_mem(out_layer->conv_bias,file);
}

void load_emb_layers(struct emb_layers * emb_layer,std::ifstream& file){
    load_to_mem(emb_layer->linear_weight,file);
    load_to_mem(emb_layer->linear_bias,file);
}

void load_in_layer(struct in_layers * in_layer,std::ifstream& file){
    load_to_mem(in_layer->group_norm_wight,file);
    load_to_mem(in_layer->group_norm_bias,file);
    load_to_mem(in_layer->conv_weight,file);
    load_to_mem(in_layer->conv_bias,file);
}

void load_ResBlock(struct ResBlock * resBlock,std::ifstream& file){
    load_in_layer(resBlock->in_layer, file);
    load_emb_layers(resBlock->emb_layer,file);
    load_out_layers(resBlock->out_layer,file);
    if(resBlock->skip_connection_weight!=NULL) load_to_mem(resBlock->skip_connection_weight,file);
    if(resBlock->skip_connection_bias!=NULL) load_to_mem(resBlock->skip_connection_bias,file);
}

void load_input_blocks(struct input_blocks* input_block,std::ifstream& file){
    load_to_mem(input_block->PaddledConv2D,file);
    load_to_mem(input_block->PaddledConv2D_bias,file);
    load_ResBlock(input_block->ResBlock1,file);
    load_SpatialTransformer(input_block->SpatialTransformer1,file);
    load_ResBlock(input_block->ResBlock2,file);
    load_SpatialTransformer(input_block->SpatialTransformer2,file);
    load_to_mem(input_block->downSample1_weight,file);
    load_to_mem(input_block->downSample1_bias,file);
    load_ResBlock(input_block->ResBlock3,file);
    load_SpatialTransformer(input_block->SpatialTransformer3,file);
    load_ResBlock(input_block->ResBlock4,file);
    load_SpatialTransformer(input_block->SpatialTransformer4,file);
    load_to_mem(input_block->downSample2_weight,file);
    load_to_mem(input_block->downSample2_bias,file);
    load_ResBlock(input_block->ResBlock5,file);
    load_SpatialTransformer(input_block->SpatialTransformer5,file);
    load_ResBlock(input_block->ResBlock6,file);
    load_SpatialTransformer(input_block->SpatialTransformer6,file);
    load_to_mem(input_block->downSample3_weight,file);
    load_to_mem(input_block->downSample3_bias,file);
    load_ResBlock(input_block->ResBlock7,file);
    load_ResBlock(input_block->ResBlock8,file);
}

void load_middle_blocks(struct middle_blocks* middle_block,std::ifstream& file){
    load_ResBlock(middle_block->ResBlock1,file);
    load_SpatialTransformer(middle_block->SpatialTransformer,file);
    load_ResBlock(middle_block->ResBlock2,file);
}

void load_output_blocks(struct output_blocks* output_block,std::ifstream& file){
    load_ResBlock(output_block->ResBlock1,file);
    load_ResBlock(output_block->ResBlock2,file);
    load_ResBlock(output_block->ResBlock3,file);
    load_to_mem(output_block->upSample1_weight,file);
    load_to_mem(output_block->upSample1_bias,file);
    load_ResBlock(output_block->ResBlock4,file);
    load_SpatialTransformer(output_block->SpatialTransformer1,file);
    load_ResBlock(output_block->ResBlock5,file);
    load_SpatialTransformer(output_block->SpatialTransformer2,file);
    load_ResBlock(output_block->ResBlock6,file);
    load_SpatialTransformer(output_block->SpatialTransformer3,file);
    load_to_mem(output_block->upSample2_weight,file);
    load_to_mem(output_block->upSample2_bias,file);
    load_ResBlock(output_block->ResBlock7,file);
    load_SpatialTransformer(output_block->SpatialTransformer4,file);
    load_ResBlock(output_block->ResBlock8,file);
    load_SpatialTransformer(output_block->SpatialTransformer5,file);
    load_ResBlock(output_block->ResBlock9,file);
    load_SpatialTransformer(output_block->SpatialTransformer6,file);
    load_to_mem(output_block->upSample3_weight,file);
    load_to_mem(output_block->upSample3_bias,file);
    load_ResBlock(output_block->ResBlock10,file);
    load_SpatialTransformer(output_block->SpatialTransformer7,file);
    load_ResBlock(output_block->ResBlock11,file);
    load_SpatialTransformer(output_block->SpatialTransformer8,file);
    load_ResBlock(output_block->ResBlock12,file);
    load_SpatialTransformer(output_block->SpatialTransformer9,file);
}

void load_unet(struct unet_model* model,std::ifstream& file){
    load_to_mem(model->time_embed_0,file);
    load_to_mem(model->time_embed_0_bias,file);
    load_to_mem(model->time_embed_2,file);
    load_to_mem(model->time_embed_2_bias,file);
    load_input_blocks(model->input_block,file);
//    load_middle_blocks(model->middle_block,file);
//    load_output_blocks(model->output_block,file);
//    load_to_mem(model->out_group_norm,file);
//    load_to_mem(model->out_group_norm_bias,file);
//    load_to_mem(model->out_corv,file);
//    load_to_mem(model->out_corv_bias,file);
}

void load_unet_from_file(struct unet_model* model,const std::string & fname) {
    std::ifstream file(fname, std::ios::binary);
    load_unet(model, file);
}

void load_to_mem(struct tensor * ptr, std::ifstream& file){
    int dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));

    // 读取每个维度的大小
    std::vector<int> sizes(dim);
    for (int i = 0; i < dim; i++) {
        file.read(reinterpret_cast<char*>(&sizes[i]), sizeof(int));
    }

    // 读取实际的float数据
    int num_elements = 1;
    //printf("shape:");
    for (int i = 0; i < dim; i++) {
        num_elements *= sizes[i];
        //printf("%d ",sizes[i]);
    }
    //printf("\n");

    int mem_size = get_tensor_size(ptr);

    if(mem_size!=num_elements){
        printf("error size %d %d\n",mem_size,num_elements);
        exit(1);
    }

    file.read(reinterpret_cast<char*>(ptr->data), num_elements * sizeof(float));
}