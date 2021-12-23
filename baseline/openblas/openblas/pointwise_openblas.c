#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

const int W = 28;
const int H = 28;
const int C = 128;
const int N = 256;
const int K = 3;

int main()
{
    clock_t start, end;
    double result;
    float a[N*C];
    float b[(W*H)*C];
    float out[N*(W*H)];

    srand(time(NULL));

    for(int i=0;i<N*C;i++)
    {
        a[i]=(float)rand()/RAND_MAX;
    }
    for(int i=0;i<C*W*H;i++)
    {
        b[i]=(float)rand()/RAND_MAX;
    }
    for(int i=0;i<N*W*H;i++)
    {
        out[i]=(float)rand()/RAND_MAX;
    }
    start = clock();
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        N,(W*H),C,1,a,C,b,C,1,out,(W*H)); 
    
    end = clock();
    result = (float)(end-start)/CLOCKS_PER_SEC;
    printf("clock = %f ms\n",result*1000);
	
    return 0;
}