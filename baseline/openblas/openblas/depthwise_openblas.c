#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#define W 7
#define H 7
#define C 1024

#define K 3
#define PAD 1
#define STRIDE 1


float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

int main()
{

clock_t start,  end;

float* in_LY;
in_LY = (float*) malloc(sizeof(float)*C*W*H);

float* filter;
filter = (float*) malloc(sizeof(float)*C*K*K);

float* output;
output = (float*) malloc(sizeof(float)*C*((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1));

float* im2col_in;
im2col_in = (float*) malloc(sizeof(float)*C*((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1)*K*K*K);

double result;

srand(time(NULL));
start = clock();

for(int i = 0; i<C ; i++)
{
     im2col_cpu(&in_LY[i*W*H], 1, H, W, K, STRIDE, PAD, &im2col_in[i*((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1)*K*K*K]);
     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 1,((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1),9,1,&filter[i*K*K],9,&im2col_in[i*((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1)*K*K*K],((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1),1,&output[i*((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1)],((W-K+2*PAD)/STRIDE+1)*((H-K+2*PAD)/STRIDE+1));
}
end = clock();
result = (float)(end-start)/CLOCKS_PER_SEC;

free(in_LY);
free(filter);
free(output);
free(im2col_in);
printf("%f ms\n", result*1000);

return 0;
}