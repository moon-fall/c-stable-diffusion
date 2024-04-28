//
// Created by 80324821 on 2023/4/18.
//
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "transformer.h"

struct tensor * new_tensor_1d(enum tensor_type type,int n0){
    struct tensor * ptensor = (struct tensor *)malloc(sizeof(struct tensor));
    ptensor->dim=1;
    ptensor->shape = (int*)malloc(sizeof(int)*1);
    ptensor->shape[0]=n0;
    ptensor->type=type;
    ptensor->data = (float*)malloc(get_type_size(type)*n0);
    return ptensor;
}

struct tensor * new_tensor_2d(enum tensor_type type,int n0,int n1){
    struct tensor * ptensor = (struct tensor *)malloc(sizeof(struct tensor));
    ptensor->dim=2;
    ptensor->shape = (int*)malloc(sizeof(int)*2);
    ptensor->shape[0]=n0;
    ptensor->shape[1]=n1;
    ptensor->type=type;
    ptensor->data = (float*)malloc(get_type_size(type)*n0*n1);
    return ptensor;
}

struct tensor * new_tensor_3d(enum tensor_type type,int n0,int n1,int n2){
    struct tensor * ptensor = (struct tensor *)malloc(sizeof(struct tensor));
    ptensor->dim=3;
    ptensor->shape = (int*)malloc(sizeof(int)*3);
    ptensor->shape[0]=n0;
    ptensor->shape[1]=n1;
    ptensor->shape[2]=n2;
    ptensor->type=type;
    ptensor->data = (float*)malloc(get_type_size(type)*n0*n1*n2);
    return ptensor;
}

struct tensor * new_tensor_4d(enum tensor_type type,int n0,int n1,int n2,int n3){
    struct tensor * ptensor = (struct tensor *)malloc(sizeof(struct tensor));
    ptensor->dim=4;
    ptensor->shape = (int*)malloc(sizeof(int)*4);
    ptensor->shape[0]=n0;
    ptensor->shape[1]=n1;
    ptensor->shape[2]=n2;
    ptensor->shape[3]=n3;
    ptensor->type=type;
    ptensor->data = (float*)malloc(get_type_size(type)*n0*n1*n2*n3);
    return ptensor;
}

struct tensor * new_tensor_5d(enum tensor_type type,int n0,int n1,int n2,int n3,int n4){
    struct tensor * ptensor = (struct tensor *)malloc(sizeof(struct tensor));
    ptensor->dim=5;
    ptensor->shape = (int*)malloc(sizeof(int)*5);
    ptensor->shape[0]=n0;
    ptensor->shape[1]=n1;
    ptensor->shape[2]=n2;
    ptensor->shape[3]=n3;
    ptensor->shape[4]=n4;
    ptensor->type=type;
    ptensor->data = (float*)malloc(get_type_size(type)*n0*n1*n2*n3*n4);
    return ptensor;
}

struct tensor * new_tensor(enum tensor_type type,int dim,int * shape){
    int size = 1;
    for(int i=0;i<dim;i++){
        size*=shape[i];
    }
    struct tensor * ptensor = (struct tensor *)malloc(sizeof(struct tensor));
    ptensor->dim=dim;
    ptensor->shape = (int*)malloc(sizeof(int)*dim);
    memcpy(ptensor->shape, shape, dim * sizeof(int));
    ptensor->type=type;
    ptensor->data = (float*)malloc(get_type_size(type)*size);
    return ptensor;
}

int get_tensor_size(struct tensor  * a){
    int len = 1;
    for(int i=0;i<a->dim;i++){
        len*=a->shape[i];
    }
    return len;
}

void tensor_set_f32(struct tensor  * a,float value){
    int len = get_tensor_size(a);
    for(int i=0;i<len;i++){
        *((float*)(a->data)+i)=value;
    }
}

void tensor_set_seq_f32(struct tensor * a){
    int len = get_tensor_size(a);
    for(int i=0;i<len;i++){
        *((float*)(a->data)+i)=(float)i;
    }
}

void tensor_set_rand_f32(struct tensor * a){
    srand(time(NULL));
    int len = get_tensor_size(a);
    for(int i=0;i<len;i++){
        *((float*)(a->data)+i)=((float) rand() / (float) RAND_MAX)-0.5;
    }
}

struct tensor * ones_tensor(int n0){
    tensor * ptensor = new_tensor_1d(GGML_TYPE_F32,n0);
    tensor_set_f32(ptensor,1.0);
    return ptensor;
}

struct tensor * ones_tensor(int n0,int n1){
    tensor * ptensor = new_tensor_2d(GGML_TYPE_F32,n0,n1);
    tensor_set_f32(ptensor,1.0);
    return ptensor;
}

struct tensor * ones_tensor(int n0,int n1,int n2){
    tensor * ptensor = new_tensor_3d(GGML_TYPE_F32,n0,n1,n2);
    tensor_set_f32(ptensor,1.0);
    return ptensor;
}

struct tensor * ones_tensor(int n0,int n1,int n2,int n3){
    tensor * ptensor = new_tensor_4d(GGML_TYPE_F32,n0,n1,n2,n3);
    tensor_set_f32(ptensor,1.0);
    return ptensor;
}

struct tensor * seq_tensor(int n0,int n1){
    tensor * ptensor = new_tensor_2d(GGML_TYPE_F32,n0,n1);
    tensor_set_seq_f32(ptensor);
    return ptensor;
}

struct tensor * seq_tensor(int n0,int n1,int n2){
    tensor * ptensor = new_tensor_3d(GGML_TYPE_F32,n0,n1,n2);
    tensor_set_seq_f32(ptensor);
    return ptensor;
}

struct tensor * seq_tensor(int n0,int n1,int n2,int n3){
    tensor * ptensor = new_tensor_4d(GGML_TYPE_F32,n0,n1,n2,n3);
    tensor_set_seq_f32(ptensor);
    return ptensor;
}

struct tensor * seq_tensor(int n0,int n1,int n2,int n3,int n4){
    tensor * ptensor = new_tensor_5d(GGML_TYPE_F32,n0,n1,n2,n3,n4);
    tensor_set_seq_f32(ptensor);
    return ptensor;
}

struct tensor * zeros_tensor(struct tensor* input){
    tensor * output;
    if(input->dim==2){
        output = new_tensor_2d(GGML_TYPE_F32,input->shape[0],input->shape[1]);
    }
    if(input->dim==3){
        output = new_tensor_3d(GGML_TYPE_F32,input->shape[0],input->shape[1],input->shape[2]);
    }
    if(input->dim==4){
        output = new_tensor_4d(GGML_TYPE_F32,input->shape[0],input->shape[1],input->shape[2],input->shape[3]);
    }
    tensor_set_f32(output,0.0);
    return output;
}

struct tensor * zeros_tensor(int n0){
    //printf("shape: %d\n",n0);
    tensor * ptensor = new_tensor_1d(GGML_TYPE_F32,n0);
    tensor_set_f32(ptensor,0.0);
    return ptensor;
}

struct tensor * zeros_tensor(int n0,int n1){
    //printf("shape: %d %d\n",n0,n1);
    tensor * ptensor = new_tensor_2d(GGML_TYPE_F32,n0,n1);
    tensor_set_f32(ptensor,0.0);
    return ptensor;
}

struct tensor * zeros_tensor(int n0,int n1,int n2){
    //printf("shape: %d %d %d\n",n0,n1,n2);
    tensor * ptensor = new_tensor_3d(GGML_TYPE_F32,n0,n1,n2);
    tensor_set_f32(ptensor,0.0);
    return ptensor;
}

struct tensor * zeros_tensor(int n0,int n1,int n2,int n3){
    //printf("shape: %d %d %d %d\n",n0,n1,n2,n3);
    tensor * ptensor = new_tensor_4d(GGML_TYPE_F32,n0,n1,n2,n3);
    tensor_set_f32(ptensor,0.0);
    return ptensor;
}

struct tensor * zeros_tensor(int n0,int n1,int n2,int n3,int n4){
    //printf("shape: %d %d %d %d %d\n",n0,n1,n2,n3,n4);
    tensor * ptensor = new_tensor_5d(GGML_TYPE_F32,n0,n1,n2,n3,n4);
    tensor_set_f32(ptensor,0.0);
    return ptensor;
}

struct tensor * zeros_tensor(int dim,int * shape){
    tensor * ptensor = new_tensor(GGML_TYPE_F32,dim,shape);
    tensor_set_f32(ptensor,0.0);
    return ptensor;
}

struct tensor * rand_tensor(int n0){
    tensor * output = new_tensor_1d(GGML_TYPE_F32,n0);
    tensor_set_rand_f32(output);
    return output;
}

struct tensor * rand_tensor(int n0,int n1){
    tensor * output = new_tensor_2d(GGML_TYPE_F32,n0,n1);
    tensor_set_rand_f32(output);
    return output;
}

struct tensor * rand_tensor(int n0,int n1,int n2){
    tensor * output = new_tensor_3d(GGML_TYPE_F32,n0,n1,n2);
    tensor_set_rand_f32(output);
    return output;
}

struct tensor * rand_tensor(int n0,int n1,int n2,int n3){
    tensor * output = new_tensor_4d(GGML_TYPE_F32,n0,n1,n2,n3);
    tensor_set_rand_f32(output);
    return output;
}

struct tensor * tensor_from_array(float *data, int n0, int n1){
    tensor * output = new_tensor_2d(GGML_TYPE_F32,n0,n1);
    int size = n0*n1;
    memcpy(output->data, data, size * sizeof(float));
    return output;
}

struct tensor * tensor_from_array(float *data,int n0,int n1,int n2){
    tensor * output = new_tensor_3d(GGML_TYPE_F32,n0,n1,n2);
    int size = n0*n1*n2;
    memcpy(output->data, data, size * sizeof(float));
    return output;
}

struct tensor * tensor_from_array(float *data,int n0,int n1,int n2,int n3){
    tensor * output = new_tensor_4d(GGML_TYPE_F32,n0,n1,n2,n3);
    int size = n0*n1*n2*n3;
    memcpy(output->data, data, size * sizeof(float));
    return output;
}

struct tensor * mm(struct tensor * input1,struct tensor * input2){
    if(input1->dim==2&&input2->dim==2){
        return mm2d(input1,input2);
    }else{
        int* mm_shape_1 = (int*)malloc(3*sizeof(int));
        int* mm_shape_2 = (int*)malloc(3*sizeof(int));
        mm_shape_1[0] = get_tensor_size(input1)/(input1->shape[input1->dim-1]*input1->shape[input1->dim-2]);
        mm_shape_1[1] = input1->shape[input1->dim-2];
        mm_shape_1[2] = input1->shape[input1->dim-1];
        mm_shape_2[0] = mm_shape_1[0];
        mm_shape_2[1] = mm_shape_1[1];
        mm_shape_2[2] = mm_shape_1[2];
        int r_dim = input1->dim;
        int dim_1=input1->dim;
        int dim_2=input2->dim;
        int* r_shape=input1->shape;
        r_shape[dim_1-1]=input2->shape[dim_2-1];
        int* shape_2=input2->shape;
        input1->dim=3;
        input2->dim=3;
        input1->shape=mm_shape_1;
        input2->shape=mm_shape_2;
        struct tensor* output =  mm3d(input1,input2);
        view(output,r_dim,r_shape);
        return output;
    }
}

struct tensor * mm2d(struct tensor * input1,struct tensor * input2){
    int M = input1->shape[0];
    int K = input1->shape[1];
    int N = input2->shape[1];
    struct tensor * output = zeros_tensor(M,N);
    float * data1 = (float*)input1->data;
    float * data2 = (float*)input2->data;
    float * res = (float*)output->data;
    for(int i=0;i<M;i++)
    {
        for(int k=0;k<K;k++)
        {
            for(int j=0;j<N;j++)
            {
                res[i*N+j]+=data1[i*K+k]*data2[k*N+j];
            }
        }
    }
    return output;
}

void mm2d(struct tensor*__restrict a,struct tensor*__restrict b,struct tensor*__restrict r){
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];
    float * adata = (float*)a->data;
    float * bdata = (float*)b->data;
    float * res = (float*)r->data;
    for(int i=0;i<M;i++)
    {
        for(int k=0;k<K;k++)
        {
            for(int j=0;j<N;j++)
            {
                res[i*N+j]+=adata[i*K+k]*bdata[k*N+j];
            }
        }
    }
    return;
}

/*
 对于3d的矩阵相乘
 第一维必须所有tensor都相等
 比如A*B=C
 A,B,C都是3维tensor A(a1,a2,a3) B(b1,b2,b3) C(c1,c2,c3)
 必须a1=b1-c1 同时剩下的维满足矩阵相乘的条件 即 a3=b2 c2=a2 c3=b3
 因此外层循环遍历a1次 而每次循环A的坐标+a2*a3 B的坐标+b2*b3 C的坐标+c2*c3
 我们用B M K N 表示 a1 a2 a3 b2
 则A+=M*K B+=K*N C+=M*N
*/

tensor* mm3d(struct tensor*__restrict a,struct tensor*__restrict b){
    int B = a->shape[0];
    int M = a->shape[1];
    int K = a->shape[2];
    int N = b->shape[2];

    tensor* r = zeros_tensor(B,M,N);

    float * adata = (float*)a->data;
    float * bdata = (float*)b->data;
    float * res = (float*)r->data;
    for(int b=0;b<B;b++){
        int r_s=b*M*N;
        int a_s=b*M*K;
        int b_s=b*K*N;
        for(int i=0;i<M;i++)
        {
            for(int k=0;k<K;k++)
            {
                for(int j=0;j<N;j++)
                {
                    res[r_s+i*N+j]+=adata[a_s+i*K+k]*bdata[b_s+k*N+j];
                }
            }
        }
    }
    return r;
}

struct tensor * mm3d(struct tensor*__restrict a,struct tensor*__restrict b,struct tensor*__restrict r){
    int B = a->shape[0];
    int M = a->shape[1];
    int K = a->shape[2];
    int N = b->shape[2];

    float * adata = (float*)a->data;
    float * bdata = (float*)b->data;
    float * res = (float*)r->data;

    for(int b=0;b<B;b++){
        int r_s=b*M*N;
        int a_s=b*M*K;
        int b_s=b*K*N;
        for(int i=0;i<M;i++)
        {
            for(int k=0;k<K;k++)
            {
                for(int j=0;j<N;j++)
                {
                    res[r_s+i*N+j]+=adata[a_s+i*K+k]*bdata[b_s+k*N+j];
                }
            }
        }
    }

    return r;
}

struct tensor* linear(struct tensor* input,struct tensor* linear){
    struct tensor* transpose_linear = transpose(linear,0,1);
    int dim = input->dim;
    int input_shape[dim];
    int output_shape[dim];
    memcpy(input_shape,input->shape,dim*sizeof(int));
    memcpy(output_shape,input->shape,dim*sizeof(int));
    output_shape[dim-1]=linear->shape[0];
    view(input, get_tensor_size(input)/input->shape[dim-1],input->shape[dim-1]);
    struct tensor* output = mm2d(input,transpose_linear);
    view(output,dim,output_shape);
    view(input,dim,input_shape);
    return output;
}

struct tensor* add(struct tensor* input1,struct tensor* input2) {
    struct tensor *output = zeros_tensor(input1);
    int size = get_tensor_size(input1);
    for (int i = 0; i < size; i++) {
        output->data[i] = input1->data[i] + input2->data[i];
    }
    return output;
}

struct tensor* add(struct tensor* input1,struct tensor* input2,bool inplace) {
    int size = get_tensor_size(input1);
    if(inplace){
        for (int i = 0; i < size; i++) {
            input1->data[i] = input1->data[i] + input2->data[i];
        }
        return input1;
    }else{
        struct tensor *output = zeros_tensor(input1);
        for (int i = 0; i < size; i++) {
            output->data[i] = input1->data[i] + input2->data[i];
        }
        return output;
    }
}

struct tensor* auto_broadcast_add(struct tensor* input1,struct tensor* input2){
    int size = get_tensor_size(input1);
    int index[input1->dim];
    for(int i=0;i<size;i++){
        int index1d = i;
        for(int j=input1->dim-1;j>=0;j--){
            index[j]=index1d%input1->shape[j];
            index1d/=input1->shape[j];
            index[j]=input2->shape[j]==1?0:index[j];
        }
        int no_broadcast_index=0;
        for(int j=0;j<input1->dim;j++){
            no_broadcast_index=no_broadcast_index*input2->shape[j]+index[j];
        }
        input1->data[i]+=input2->data[no_broadcast_index];
    }
    return input1;
}

struct tensor* minus(struct tensor* input1,struct tensor* input2) {
    struct tensor *output = zeros_tensor(input1);
    int size = get_tensor_size(input1);
    for (int i = 0; i < size; i++) {
        output->data[i] = input1->data[i] - input2->data[i];
    }
    return output;
}

struct tensor* multiply(struct tensor* input1,struct tensor* input2,bool inplace){
    int size = get_tensor_size(input1);
    if(inplace){
        for (int i = 0; i < size; i++) {
            input1->data[i] = input1->data[i] * input2->data[i];
        }
        return input1;
    }else{
        struct tensor *output = zeros_tensor(input1);
        for (int i = 0; i < size; i++) {
            output->data[i] = input1->data[i] * input2->data[i];
        }
        return output;
    }
}

struct tensor* divide(struct tensor* input1,struct tensor* input2) {
    struct tensor *output = zeros_tensor(input1);
    int size = get_tensor_size(input1);
    for (int i = 0; i < size; i++) {
        output->data[i] = input1->data[i] / input2->data[i];
    }
    return output;
}

struct tensor* mean_last_dim(struct tensor* input)
{
    int last_dim = input->shape[input->dim-1];
    float sum=0;
    int size = get_tensor_size(input);
    struct tensor* output = zeros_tensor(input);
    for(int i=0;i<size;i++){
        sum += input->data[i];
        if(i%last_dim==last_dim-1){
            for(int j=i;j>i-last_dim;j--){
                output->data[j] = sum / last_dim;
            }
            sum=0;
        }
    }
    return output;
}

struct tensor* var_last_dim(struct tensor* input,struct tensor* mean){
    int last_dim = input->shape[input->dim-1];
    float sum=0;
    int size = get_tensor_size(input);
    struct tensor* output = zeros_tensor(input);
    for(int i=0;i<size;i++){
        sum += (input->data[i]-mean->data[i])*(input->data[i]-mean->data[i]);
        //printf("i:%d sum:%f\n",i,sum);
        if(i%last_dim==last_dim-1){
            for(int j=i;j>i-last_dim;j--){
                output->data[j] = sum / (last_dim-1);
            }
            sum=0;
        }
    }
    return output;
}

struct tensor* std_last_dim(struct tensor* input,struct tensor* mean){
    int last_dim = input->shape[input->dim-1];
    float sum=0;
    int size = get_tensor_size(input);
    struct tensor* output = zeros_tensor(input);
    for(int i=0;i<size;i++){
        sum += (input->data[i]-mean->data[i])*(input->data[i]-mean->data[i]);
        //printf("i:%d sum:%f\n",i,sum);
        if(i%last_dim==last_dim-1){
            float std = sqrt(sum / (last_dim));
            for(int j=i;j>i-last_dim;j--){
                output->data[j] = std;
            }
            sum=0;
        }
    }
    return output;
}

struct tensor* layer_norm(struct tensor* input) {
    tensor* mean = mean_last_dim(input);
    tensor* std = std_last_dim(input,mean);
    tensor* output = divide(minus(input,mean),tensor_scaled_add(std,0.00001));
    return output;
}

struct tensor* group_norm_4d(int num_groups ,struct tensor* input) {
    int n0 = input->shape[0];
    int n1 = input->shape[1];
    int n2 = input->shape[2];
    int n3 = input->shape[3];
    view(input,n0,num_groups,n1/num_groups * n2 * n3);
    tensor* mean = mean_last_dim(input);
    tensor* std = std_last_dim(input,mean);
    tensor* output = divide(minus(input,mean),tensor_scaled_add(std,0.00001));
    view(input,n0,n1,n2,n3);
    view(output,n0,n1,n2,n3);
    return output;
}

struct tensor* view(struct tensor* t,int n0,int n1,int n2,int n3){
    free(t->shape);
    t->shape = (int*)malloc(sizeof(int)*4);
    t->shape[0]=n0;
    t->shape[1]=n1;
    t->shape[2]=n2;
    t->shape[3]=n3;
    t->dim=4;
    return t;
}

struct tensor* view(struct tensor* t,int n0,int n1,int n2){
    free(t->shape);
    t->shape = (int*)malloc(sizeof(int)*3);
    t->shape[0]=n0;
    t->shape[1]=n1;
    t->shape[2]=n2;
    t->dim=3;
    return t;
}

struct tensor* view(struct tensor* t,int n0,int n1){
    free(t->shape);
    t->shape = (int*)malloc(sizeof(int)*2);
    t->shape[0]=n0;
    t->shape[1]=n1;
    t->dim=2;
    return t;
}

struct tensor* view(struct tensor* t,int n0){
    free(t->shape);
    t->shape = (int*)malloc(sizeof(int)*1);
    t->shape[0]=n0;
    t->dim=1;
    return t;
}

struct tensor* view(struct tensor* t,int dim,int* shape){
    free(t->shape);
    t->dim=dim;
    t->shape=(int*)malloc(dim*sizeof(int));
    memcpy(t->shape,shape,dim*sizeof(int));
    return t;
}

/*
对于2d转置 首先我们需要将维度的元素个数调换
其次需要将元素调换
比如对于tensor(2,3) transpose(0,1)
我们需要将tensor变为tensor(3,2)
而每个元素比如坐标为(x,y)的元素需要变到(y,x)的位置
比如[[1,2,3]
     [4,5,6]]
变化为[[1,4],
       [2,5],
       [3,6]]
看3这个元素 原始的2维度坐标是(0,2) 1维坐标是 0*3+2=2
变化后 2维度坐标是(2,0) 1维坐标是 2*2+0=4

因此一个tensor(n0,n1)转置后在(x,y)的元素转置后在(y,x)
其一维坐标由x*n0+y变为y*n1+x

如果是3维tensor 变化后两维 则同样第一维度只是一个外层循环

如果是4维tensor 变化中间两维 则同样第一维度只是一个外层循环
设维度为(I,J,K,L) 变化后维度为 (I,K,J,L)
同时(i,j,k,l)变为(i,k,j,l)


对于通用transpose我们还是需要将tensor看成一维的
然后根据维度信息 求出原始的多维坐标 然后变化为新的多维坐标
在转到一维上
*/

struct tensor* transpose(struct tensor* input,int dim1,int dim2){
    int size = get_tensor_size(input);
    int index[input->dim];
    int out_shape[input->dim];
    memcpy(out_shape, input->shape, input->dim * sizeof(int));

    int tmp=out_shape[dim1];
    out_shape[dim1]=out_shape[dim2];
    out_shape[dim2]=tmp;

    struct tensor* output = zeros_tensor(input->dim,out_shape);

    float* p = (float*)input->data;
    for(int i=0;i<size;i++){
        int srcindex1d=i;
        for(int d=input->dim-1;d>=0;d--){
            index[d]=srcindex1d%input->shape[d];
            srcindex1d/=input->shape[d];
        }
        int tmp = index[dim1];
        index[dim1]=index[dim2];
        index[dim2]=tmp;
        int index1d=0;
        int multi=1;
        for(int d=input->dim-1;d>=0;d--){
            index1d+=index[d]*multi;
            multi*=out_shape[d];
        }
        output->data[index1d]=p[i];
    }
    return output;
}

tensor* tensor_scaled_division(tensor* t,float div){
    int size = get_tensor_size(t);
    float* p = (float*)t->data;
    for(int i=0;i<size;i++){
        p[i]/=div;
    }
    return t;
}

tensor* tensor_scaled_add(tensor* t,float value){
    int size = get_tensor_size(t);
    float* p = (float*)t->data;
    for(int i=0;i<size;i++){
        p[i]+=value;
    }
    return t;
}

void softmax_last_dim(struct tensor* input){
    int last_dim = input->shape[input->dim-1];
    float sum=0;
    int size = get_tensor_size(input);
    for(int i=0;i<size;i++){
        input->data[i]=exp(input->data[i]);
        sum += input->data[i];
        if(i%last_dim==last_dim-1){
            for(int j=i;j>i-last_dim;j--){
                input->data[j] = input->data[j] / sum;
            }
        }
    }
    return;
}

float relu(float x) {
    return x < 0 ? 0 : x;
}

void relu_tensor(tensor *input) {
    int size = get_tensor_size(input);
    for (int i = 0; i < size; ++i) {
        input->data[i] = relu(input->data[i]); // 对每个元素执行ReLU
    }
}

float gelu(float x) {
    float pi = 3.14159265358979323846;
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * pow(x, 3))));
}

void gelu_tensor(tensor *input) {
    int size = get_tensor_size(input);
    for (int i = 0; i < size; ++i) {
        input->data[i] = gelu(input->data[i]);
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void silu_tensor(tensor *input) {
    int size = get_tensor_size(input);
    for (int i = 0; i < size; ++i) {
        input->data[i] = sigmoid(input->data[i]);
    }
}

struct tensor* cat(struct tensor* input1,struct tensor* input2,int dim){
    int* shape = (int*) malloc(input1->dim*sizeof(int));
    memcpy(shape,input1->shape,input1->dim*sizeof(int));
    shape[dim]+=input2->shape[dim];
    struct tensor* output = zeros_tensor(input1->dim,shape);
    int common_dim_size = 1;
    for(int i=0;i<dim;i++){
        common_dim_size *= input1->shape[i];
    }
    int input1_dim_size = 1;
    int input2_dim_size = 1;
    int r_dim_size = 1;
    for(int i=0;i<input1->dim;i++){
        r_dim_size*=(input1->shape[i]+input2->shape[i]);
        input1_dim_size*=input1->shape[i];
        input2_dim_size*=input2->shape[i];
    }
    for(int i=0;i<common_dim_size;i++){
        int r_index=i*r_dim_size;
        int input1_index=i*input1_dim_size;
        int input2_index=i*input2_dim_size;
        memcpy(output->data+r_index,input1->data+input1_index,input1_dim_size*sizeof(int));
        memcpy(output->data+r_index+input1_dim_size,input2->data+input2_index,input2_dim_size*sizeof(int));
    }
    return output;
}

float sum(struct tensor* input){
    int size = get_tensor_size(input);
    float sum=0.0;
    for(int i=0;i<size;i++){
        sum+=input->data[i];
    }
    return sum;
}


int get_type_size(enum tensor_type type){
    switch(type){
        case GGML_TYPE_I8:
            return 1;
            break;
        case GGML_TYPE_I16:
            return 2;
            break;
        case GGML_TYPE_I32:
            return 4;
            break;
        case GGML_TYPE_F16:
            return 2;
            break;
        case GGML_TYPE_F32:
            return 4;
            break;
    }
    return 0;
}

void test_mm2d(){
    tensor* a = ones_tensor(1024,1024);
    tensor* b = ones_tensor(1024,1024);
    tensor* r = zeros_tensor(1024,1024);
    clock_t start = clock();
    mm2d(a,b,r);
    clock_t end = clock();
    printf("cost %lf ms\n",(double)(end-start)/1000);
    for(int i=0;i<100;i++){
        float f = ((float*)(r->data))[i];
        printf("%f ",f);
    }
    printf("\n");
}

void test_transpose_3d(){
    tensor* a = seq_tensor(2,3,4);
    tensor_print(a);
    struct tensor* b = transpose(a,0,1);
    tensor_print(b);
    return;
}

void test_transpose_4d(){
    tensor* a = seq_tensor(2,3,4,5);
    tensor_print(a);
    transpose(a,0,3);
    tensor_print(a);
    return;
}

void test_softmax(){
    tensor* a = seq_tensor(2,3);
    tensor_print(a);
    softmax_last_dim(a);
    tensor_print(a);
    return;
}

void test_layer_norm(){
    tensor* a  = seq_tensor(2,3);
    tensor_print(a);
    a=layer_norm(a);
    tensor_print(a);
    return;
}

void test_group_norm(){
    tensor* a  = seq_tensor(1,6,2, 1);
    tensor_print(a);
    a=group_norm_4d(3,a);
    tensor_print(a);
    return;
}

void test_cat(){
    struct tensor* x1 = seq_tensor(2,3);
    struct tensor* x2 = seq_tensor(2,3);
    struct tensor* r = cat(x1,x2,1);
    tensor_print(r);
}

void shape_print(tensor* a) {
    printf("dim:");
    for(int i=0;i<a->dim;i++){
        printf("%d ",a->shape[i]);
    }
    printf("\n");
    fflush(stdout);
}

void tensor_print(tensor* a){
    int size = get_tensor_size(a);
    int dim_product[a->dim];
    dim_product[a->dim-1]=a->shape[a->dim-1];
    for(int i=a->dim-2;i>=0;i--){
        dim_product[i]=dim_product[i+1]*a->shape[i];
    }
    float* f = (float*)a->data;

    printf("dim:");
    for(int i=0;i<a->dim;i++){
        printf("%d ",a->shape[i]);
    }
    printf("\n");
    for(int i=0;i<size;i++){
        for(int j=0;j<a->dim;j++){
            if(i%dim_product[j]==0&&i!=0){
                printf("\n");
            }
        }
        printf("%f ",f[i]);
    }
    printf("\n");
    return;
}

int main(){
    test_transpose_4d();
}

