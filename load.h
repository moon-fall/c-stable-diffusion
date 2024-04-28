//
// Created by 80324821 on 2023/4/24.
//

#ifndef UNTITLED_LOAD_H
#define UNTITLED_LOAD_H

#include <fstream>

void load();
void load_to_mem(struct tensor* ptr, std::ifstream& file);
void load_unet_from_file(struct unet_model* model,const std::string & fname);

#endif //UNTITLED_LOAD_H
