#ifndef _CNN_H
#define _CNN_H

typedef float DTYPE;                                                                                             

const int W = 30;
const int H = 6;
const int C = 32;
const int K = 3;
#define PAD 1
#define STRIDE 1

#define Tw 16
#define Tc 4
void depthwise_conv(
        DTYPE* in0,
          //hls::vector<DTYPE, 16>* in1,
          //hls::vector<DTYPE, 16>* in2,
          //hls::vector<DTYPE, 16>* in3,       
          DTYPE* kernel,
          DTYPE* out,
		  //hls::vector<DTYPE, 16>* out1,
		  //hls::vector<DTYPE, 16>* out2,
		  //hls::vector<DTYPE, 16>* out3,
          int W,
          int H,
          int C,
          int K,
          int pad,
          int stride,//,
          int num_nonzero,
          int iter
         // int Tw,
         // int Tc
                    );
#endif 
