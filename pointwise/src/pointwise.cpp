#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
//#include "assert.h"

#include "pointwise.h"
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

extern "C" {

void readA(hls::vector<DTYPE, BUSWIDTH> *A, hls::stream<hls::vector<DTYPE, BUSWIDTH> > & AStream , int W, int H, int C, int N, int wh_bound, bool mode)
{
 int N_BM = N/BLOCK_M;
 int C_16 = C/16;
 
	for(int ib = 0; ib < N_BM; ib++){
		for(int jb = 0; jb < wh_bound; jb++){
			for(int kb = 0; kb < C_16; kb++){
				for(int i = 0; i < BLOCK_M; i++){
					for(int k = 0; k < 1; k++){ //k < BLOCK_K/16
						#pragma HLS pipeline II=1
						hls::vector<DTYPE, BUSWIDTH> A_temp = A[((ib*BLOCK_M+i)*(C_16) + kb +k)]; // kb*(BLOCK_K/16), C/16
						AStream.write(A_temp);
					}
				}
			}
		}
	}

	if(mode)
	{
		for(int ib = 0; ib < N_BM; ib++){
				for(int kb = 0; kb < C_16; kb++){
					for(int i = 0; i < BLOCK_M; i++){
						for(int k = 0; k < 1; k++){ //k < BLOCK_K/16
							#pragma HLS pipeline II=1
							hls::vector<DTYPE, BUSWIDTH> A_temp = A[((ib*BLOCK_M+i)*(C_16) + kb +k)];  //kb*(BLOCK_K/16)
							AStream.write(A_temp);
						}
					}
				}
		}
	}
}

void readBt(hls::vector<DTYPE, BUSWIDTH> *Bt, hls::stream<hls::vector<DTYPE, BUSWIDTH> > & BStream , int W, int H, int C, int N, int wh_bound, bool wh_mode, int wh_zero)
{
	DTYPE* B_DTYPE = (DTYPE*) Bt;
	hls::vector<DTYPE, BUSWIDTH> B_temp;
  int C_16 = C/16;
  int N_BM = N/BLOCK_M;

	for(int ib = 0; ib < N_BM; ib++) {
		for(int jb = 0; jb < wh_bound; jb++) {
			for(int kb = 0; kb < C_16; kb++) {
				for(int j = 0; j < BLOCK_N; j++) {
				  	for(int k = 0; k < 1; k++) {  //k < BLOCK_K/16
						#pragma HLS pipeline II=1
          				B_temp = Bt[((jb*BLOCK_N+j)*(C_16) + kb + k)]; //kb*(BLOCK_K/16)
         				BStream.write(B_temp);
					}	
				}
			}
		}
	} 

	if(wh_mode)
	{
		for(int ib = 0; ib < N_BM; ib++) {
			for(int kb = 0; kb < C_16; kb++) {
				for(int j = 0; j < BLOCK_N; j++) {
					for(int k = 0; k < BLOCK_K; k+=16) {
						#pragma HLS pipeline II=1
						hls::vector<DTYPE, BUSWIDTH>* temp_ptr =(hls::vector<DTYPE, BUSWIDTH>*)( B_DTYPE + ((wh_bound*BLOCK_N+j-wh_zero)*C + kb*(BLOCK_K) + k));
						B_temp = *temp_ptr;
						BStream.write(B_temp);
					}	
				}
			}
		} 
	}
}

void pointwise_conv(hls::stream<hls::vector<DTYPE, BUSWIDTH> > &AStream, hls::stream<hls::vector<DTYPE, BUSWIDTH> > &BStream, hls::vector<DTYPE, BUSWIDTH> *AB, int W, int H, int C, int N,int wh_bound, bool wh_mode, int wh_zero)
{

	hls::vector<DTYPE, BUSWIDTH>* temp_ptr;
	hls::vector<DTYPE, BUSWIDTH> A_block[BLOCK_M] = {0};
	hls::vector<DTYPE, BUSWIDTH> B_block[BLOCK_N] = {0};
	hls::vector<DTYPE, BUSWIDTH> AB_block[BLOCK_M] = {0};
	DTYPE* AB_DTYPE = (DTYPE*)AB;
    int N_BM = N/BLOCK_M;
	int C_16 = C/16;
    int WH = W*H;
   // bool wh_mode = ((W*H) % BLOCK_N == 0) ? 0 : 1;
   // int wh_bound = (W*H)/BLOCK_N ;
   // int wh_zero = 16 -((W*H) - wh_bound*BLOCK_N);

	for(int ib = 0; ib < N_BM; ib++) 
	{
 		for(int jb = 0; jb < wh_bound; jb++)
		{
			for(int t = 0; t < BLOCK_M; t++) 
			{
       			#pragma HLS UNROLL
				AB_block[t] = 0;			
			}

			for(int kb = 0; kb < C_16; kb++) 
			{
				for(int i = 0; i < BLOCK_M; i++) 
				{	
					#pragma HLS pipeline II=1	
					for(int k = 0; k < 1; k++)// k < BLOCK_K/BUSWIDTH
					#pragma HLS UNROLL
					{		
						A_block[i] = AStream.read();
						B_block[i] = BStream.read();
					}				
				}


				for(int m = 0; m < BLOCK_M; m++) 
				{
					#pragma HLS pipeline II=1 		
					for(int n = 0; n < BLOCK_N; n++) 
					{	
						for(int k = 0; k < BLOCK_K; k++ )
						{
							AB_block[m][n] +=  A_block[m][k]*B_block[n][k];
						}
					}				
				}
			}						

			for(int i = 0; i < BLOCK_M; i++)
			{
				#pragma HLS pipeline II=1
     				 temp_ptr = (hls::vector<DTYPE, BUSWIDTH> *)(AB_DTYPE + ((ib*BLOCK_M + i)*(WH) + jb*BLOCK_N));
				*temp_ptr = AB_block[i];
				/*for(int t = 0; t < BUSWIDTH; t++)
				{
					*(AB + ((ib*BLOCK_M + i)*(WH) + jb*BLOCK_N)) = AB_BLOCK[i][t];

				}*/
			}
		}
	}

	if(wh_mode)		
	{
		for(int ib = 0; ib < N_BM; ib++)
		{
     
      for(int i = 0; i < BLOCK_M; i++) 
			{	    
          AB_block[i] = 0;
			} 
			for(int kb = 0; kb < C_16; kb++) 
			{
				for(int i = 0; i < BLOCK_M; i++) 
				{		
			      //������ k for loop �־����.
						#pragma HLS pipeline II=1
						A_block[i] = AStream.read();
						B_block[i] = BStream.read();						
				}

				for(int m = 0; m < BLOCK_M; m++) 
				{
				#pragma HLS pipeline II=1 		
					for(int n = 0; n < BLOCK_M; n++) 
					{	
						for(int k = 0; k < BLOCK_K; k++ )
						{
							AB_block[m][n] += A_block[m][k]*B_block[n][k];
						}
					}				
				}
			}					
			for(int i = 0; i < BLOCK_M; i++)
			{
				#pragma HLS pipeline II=1
				temp_ptr = (hls::vector<DTYPE, BUSWIDTH> *)(AB_DTYPE + ((ib*BLOCK_M + i)*(WH) + wh_bound*BLOCK_N - wh_zero));
				*temp_ptr = AB_block[i];
			}
		}
		
	}
}

void pointwise(hls::vector<DTYPE, BUSWIDTH> * A, hls::vector<DTYPE, BUSWIDTH> * Bt, hls::vector<DTYPE, BUSWIDTH> * AB, int W, int H, int C, int N)
{
	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 port=A
	#pragma HLS INTERFACE mode=m_axi bundle=gmem1 port=Bt
	#pragma HLS INTERFACE mode=m_axi bundle=gmem2 port=AB

	#pragma HLS DATAFLOW

	bool wh_mode = ((W*H) % BLOCK_N == 0) ? 0 : 1;
	int wh_bound = (W*H)/BLOCK_N ;
	int wh_zero = 16 -((W*H) - wh_bound*BLOCK_N);

	hls::stream<hls::vector<DTYPE, BUSWIDTH> > AStream("AStream");
	hls::stream<hls::vector<DTYPE, BUSWIDTH> > BStream("BStream");

	readA(A, AStream, W, H, C, N, wh_bound, wh_mode);
	readBt(Bt, BStream, W, H, C, N, wh_bound, wh_mode, wh_zero);
	pointwise_conv(AStream, BStream, AB, W, H, C, N,  wh_bound, wh_mode, wh_zero);
}
}



