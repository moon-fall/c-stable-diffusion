//
// Created by 80324821 on 2023/4/30.
//

#include <stdio.h>

#include "main.h"
#include "unet.h"
#include "load.h"
#include "tensor.h"



//int main() {
//    struct unet_model* unet = init_unet_model();
//    printf("finish init\n");
//    load_unet_from_file(unet,"D:\\src\\stable-diffusion\\my_stable_diffusion\\tensor_data.bin");
//
//    struct tensor * latent = rand_tensor(1,4,64,64);
//    struct tensor * text_embeddings = rand_tensor(1, 77, 768);
//
//
//    struct tensor * output = unet_forward(latent,text_embeddings,0,unet);
//    tensor_print(output);
//};
