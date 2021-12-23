#ifndef _MM_H
#define _MM_H

typedef float DTYPE;

#define N_SIZE 64   //number of filter
#define W_SIZE 112	  //input width
#define H_SIZE 112 //input height
#define C_SIZE 32   //number of channel

const int BLOCK_SIZE = 16;
const int BUSWIDTH = 16;

void pointwise(float *A,
               float *Bt,
               float *AB, 
               int W,
               int H,
               int C,
               int N
               );
#endif
