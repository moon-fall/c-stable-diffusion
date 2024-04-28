//
// Created by 80324821 on 2023/4/18.
//

#ifndef UNTITLED_TENSOR_H
#define UNTITLED_TENSOR_H

enum tensor_type {
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_F16,
    GGML_TYPE_F32
};

struct tensor{
    int dim;
    int* shape;
    enum tensor_type type;
    float* data;
};

struct tensor* transpose(struct tensor* t,int dim1,int dim2);
struct tensor* view(struct tensor* t,int n0);
struct tensor*  view(struct tensor* t,int n0,int n1);
struct tensor*  view(struct tensor* t,int n0,int n1,int n2);
struct tensor*  view(struct tensor* t,int n0,int n1,int n2,int n3);
struct tensor* view(struct tensor* t,int dim,int* shape);
struct tensor * mm2d(struct tensor * a,struct tensor * b);
struct tensor * mm3d(struct tensor * a,struct tensor * b);
inline void mm2d(struct tensor*__restrict a,struct tensor*__restrict b,struct tensor*__restrict r);
struct tensor* linear(struct tensor* input,struct tensor* linear);
int get_type_size(enum tensor_type type);
struct tensor * new_tensor_1d(enum tensor_type type,int n0);
struct tensor * new_tensor_2d(enum tensor_type type,int n0,int n1);
struct tensor * new_tensor_3d(enum tensor_type type,int n0,int n1,int n2);
void tensor_set_f32(struct tensor  * a,float value);
struct tensor * zeros_tensor(int n0);
struct tensor * zeros_tensor(int n0,int n1);
struct tensor * zeros_tensor(int n0,int n1,int n2);
struct tensor * zeros_tensor(int n0,int n1,int n2,int n3);
struct tensor * zeros_tensor(int n0,int n1,int n2,int n3,int n4);
struct tensor * zeros_tensor(int dim,int * shape);
struct tensor * new_tensor(enum tensor_type type,int dim,int * shape);
struct tensor * ones_tensor(int n0);
struct tensor * ones_tensor(int n0,int n1);
struct tensor * ones_tensor(int n0,int n1,int n2);
struct tensor * ones_tensor(int n0,int n1,int n2,int n3);
struct tensor * seq_tensor(int n0,int n1);
struct tensor * seq_tensor(int n0,int n1,int n2);
struct tensor * seq_tensor(int n0,int n1,int n2,int n3);
struct tensor * seq_tensor(int n0,int n1,int n2,int n3,int n4);
struct tensor * rand_tensor(int n0);
struct tensor * rand_tensor(int n0,int n1);
struct tensor * rand_tensor(int n0,int n1,int n2);
struct tensor * rand_tensor(int n0,int n1,int n2,int n3);
struct tensor * tensor_from_array(float *data, int n0, int n1);
struct tensor * tensor_from_array(float *data,int n0,int n1,int n2);
struct tensor * tensor_from_array(float *data,int n0,int n1,int n2,int n3);
tensor* tensor_scaled_division(tensor* t,float div);
tensor* tensor_scaled_add(tensor* t,float value);
float gelu(float x);
float relu(float x);
void relu_tensor(tensor *input);
void silu_tensor(tensor *input);
struct tensor* add(struct tensor* input1,struct tensor* input2);
struct tensor* add(struct tensor* input1,struct tensor* input2,bool inplace);
struct tensor* auto_broadcast_add(struct tensor* input1,struct tensor* input2);
void softmax_last_dim(struct tensor* input);
struct tensor* layer_norm(struct tensor* input);
struct tensor* group_norm_4d(int num_groups ,struct tensor* input);
void shape_print(tensor* a);
void tensor_print(tensor* a);
void shape_print(tensor* a);
int get_tensor_size(struct tensor  * a);
struct tensor* cat(struct tensor* input1,struct tensor* input2,int dim);
float sum(struct tensor* input);

struct tensor * mm(struct tensor * a,struct tensor * b);

#endif //UNTITLED_TENSOR_H
