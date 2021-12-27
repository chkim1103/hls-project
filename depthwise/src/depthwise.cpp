#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"

#include <iostream>
#include "depthwise.h"

//VER1. H ?? PIPELINE??? ??. channel? ?? ?? -> reg ?? ?? version

void load_input(       hls::vector<DTYPE, Tw>* in,
                       hls::stream<hls::vector<DTYPE, Tw> >& inStream,
                       int W,
                       int H,
                       int C,
                       int k,
                       int pad,
                       int stride,
                       int num_nonzero
                       ) 
{
  DTYPE* in_dtype = (DTYPE*)in;
//  char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? Tw-2 : ((W-K+2*pad)+1)%(Tw-2);
//  char num_nonzero = temp + 2 - pad;
  static bool mask_nonzero[Tw] = {0};
  hls::vector<DTYPE, Tw> temp_row = 0;
 	#pragma HLS ARRAY_PARTITION variable=mask_nonzero complete dim=0 
  
  int WB_1 = ((W-K+2*pad)+1-1)/(Tw-2) + 1;
  int WB = ((W-K+2*pad)+1-1)/(Tw-2);
  int WH = W*H;
  int TW_2 = Tw-2;
  
  for(int i = 0; i < Tw; i++)
  {  
      #pragma HLS UNROLL
      mask_nonzero[i] = ((num_nonzero-i)>0) ? 1:0; 
  }
	for(int ct = 0; ct < C; ct += Tc)
	{
		for(int wb = 0; wb < WB_1; wb++)    
		{
			if(pad)
			{
				for(int c = 0; c < Tc; c += 1)
				{
				#pragma HLS pipeline II=1
					temp_row = 0;
					inStream.write(temp_row);
				}
			}
			
			for(int h = 0; h < H; h++) 
			{
  				for(int c = 0; c < Tc; c += 1)
  				{
  					#pragma HLS pipeline II=1
  					hls::vector<DTYPE, Tw>* in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((c + ct)*WH + h*W + wb*(TW_2) - pad));
  					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
  					temp_row[0] = (pad && (wb==0)) ? 0 : temp_row[0];
  					
  					if(wb == WB)
  					{
  						for(int i = 0; i < Tw; i++)
  						{
  							temp_row[i] = (mask_nonzero[i]) ?  temp_row[i] : 0;
  						}
  					}

  					inStream.write(temp_row);
	 	  		}
   			} 
			
				if(pad)
				{
					for(int c = 0; c < Tc; c += 1)
					{
					#pragma HLS pipeline II=1
						temp_row = 0;
						inStream.write(temp_row);
					}
				}
				
		}
	}
}


void depthwise_conv(hls::stream<hls::vector<DTYPE, Tw> >& in0_stream,
                    //hls::stream<hls::vector<DTYPE, 16> >& in1_stream,
                    //hls::stream<hls::vector<DTYPE, 16> >& in2_stream,                     
                    //hls::stream<hls::vector<DTYPE, 16> >& in3_stream,
                    hls::vector<DTYPE, 9>* kernel,                                        
                    hls::stream<hls::vector<DTYPE, (Tw-2)> >&out0_stream, 
				//	hls::stream<hls::vector<DTYPE, 4> >&out1_stream, 
				//	hls::stream<hls::vector<DTYPE, 4> >&out2_stream, 
				//	hls::stream<hls::vector<DTYPE, 4> >&out3_stream, 
                    int W,
                    int H,
                    int C,
                    int k,
                    int pad,
                    int stride,
                    int num_nonzero,
                    int iter
                    )
{
    
    static DTYPE kernel_buf[Tc][3][3] = {0};
    static DTYPE output_buf[Tc][2][Tw-2] = {0};
    static DTYPE output_temp[Tc][2][Tw-2]= {0};
    hls::vector<DTYPE, (Tw-2)> store_output = 0;
    hls::vector<DTYPE, 9> kernel_temp = 0;

	#pragma HLS ARRAY_PARTITION variable=kernel_buf complete dim=0
	#pragma HLS ARRAY_PARTITION variable=output_buf complete dim=0
	#pragma HLS ARRAY_PARTITION variable=output_temp complete dim=0  
	
 int WB_1 = ((W-K+2*pad)+1-1)/(Tw-2) + 1;
 int WB   = ((W-K+2*pad)+1-1)/(Tw-2);
 int H_2PAD = H+ 2*pad;
 
     
   
  for(int ct = 0; ct < C; ct += Tc)
	{  
    for(int i = 0; i < Tc; i++)
    {
        //#pragma HLS UNROLL		
		    #pragma HLS pipeline II = 1
         kernel_temp = kernel[i + ct];
        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < K; k++)
            {
			//#pragma HLS LOOP_TRIPCOUNT min = 3 max = 3
                kernel_buf[i][j][k] = kernel_temp[j*K + k];//kernel[i*K*K + K*j + k];
            }
        }
    }
     
		for(int wb = 0; wb < WB_1 ; wb++)
		{	
			if(wb == WB)
			{
				for(int h = 0; h < H_2PAD ; h++)     
				{  
					for(int c = 0; c < Tc ; c++)
					{  
						#pragma HLS pipeline II = 1     
						hls::vector<DTYPE, Tw> input_temp0 = in0_stream.read();	
						
						for(int w = 0; w < 14 ; w++)  //???? ITER (iter)
						{
						//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
							output_buf[c][0][w] = input_temp0[stride*w]*kernel_buf[c][0][0]+input_temp0[stride*w+1]*kernel_buf[c][0][1]+input_temp0[stride*w+2]*kernel_buf[c][0][2];
							output_buf[c][1][w] = input_temp0[stride*w]*kernel_buf[c][1][0]+input_temp0[stride*w+1]*kernel_buf[c][1][1]+input_temp0[stride*w+2]*kernel_buf[c][1][2]+output_temp[c][0][w];
							store_output[w] = input_temp0[stride*w]*kernel_buf[c][2][0]+input_temp0[stride*w+1]*kernel_buf[c][2][1]+input_temp0[stride*w+2]*kernel_buf[c][2][2]+output_temp[c][1][w];
													
							output_temp[c][0][w] = output_buf[c][0][w];
							output_temp[c][1][w] = output_buf[c][1][w];					
						}									
						if(h>1&&(h%stride==0))
						{   								
							out0_stream.write(store_output);	
						}
					}             
				}	
			}
			
			//#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1    
			else
			{
				for(int h = 0; h < H_2PAD ; h++)     
				{   
					for(int c = 0; c < Tc ; c++)
					{  
						#pragma HLS pipeline II = 1     
						hls::vector<DTYPE, Tw> input_temp0 = in0_stream.read();												
					
						for(int w = 0; w < 14 ; w++)  //pipeline   ???? (Tw-2)/stride
						{                                                
						//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
							output_buf[c][0][w] = input_temp0[stride*w]*kernel_buf[c][0][0]+input_temp0[stride*w+1]*kernel_buf[c][0][1]+input_temp0[stride*w+2]*kernel_buf[c][0][2];
							output_buf[c][1][w] = input_temp0[stride*w]*kernel_buf[c][1][0]+input_temp0[stride*w+1]*kernel_buf[c][1][1]+input_temp0[stride*w+2]*kernel_buf[c][1][2]+output_temp[c][0][w];
							store_output[w] = input_temp0[stride*w]*kernel_buf[c][2][0]+input_temp0[stride*w+1]*kernel_buf[c][2][1]+input_temp0[stride*w+2]*kernel_buf[c][2][2]+output_temp[c][1][w];
													
							output_temp[c][0][w] = output_buf[c][0][w];
							output_temp[c][1][w] = output_buf[c][1][w];					
						}	
						if(h>1&&((h )%stride==0))
						{   								
							out0_stream.write(store_output);	
						}					
					}        							
				}
			}
		}
	}
    // for wb = WB
      
}

void store_output(    hls::stream<hls::vector<DTYPE, (Tw-2)> >& out_stream,				  
                      hls::vector<DTYPE, Tc>* out,
                      int W,
                      int H,
                      int C,
                      int k,
                      int pad,
                      int stride,
                    char num_nonzero,
                    char iter	)
{  

 int WB_1 = ((W-K+2*pad)+1-1)/(Tw-2) + 1;
 int WB   = ((W-K+2*pad)+1-1)/(Tw-2);
 int WK   = ((W-K+2*pad)/stride+1);
   int H_OUT = ((H-K+ 2*pad)/stride+1);
 int CTC = C/Tc;
 int TW_2_STRIDE = (Tw-2)/stride;

static hls::vector<DTYPE, Tw-2> output_temp = 0;
static hls::vector<DTYPE, Tc> output_buf0[Tw-2] = {0};


static hls::vector<DTYPE, Tc> test = 0;

  	for(int ct = 0; ct < C; ct += Tc)
  	{
    		for(int wb = 0; wb < WB_1; wb++) 
    		{
      			for(int h = 0; h < H_OUT; h++) 
      			{		
					    if(wb == WB)	
			    		{ 
		    				for(int c = 0; c < Tc; c++) //pipeline
		    				{	
      							#pragma HLS pipeline II = 1
      							if(h != H_OUT)
      							{
      								output_temp = out_stream.read();
      							}       						
      								for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
      								{
      									output_buf0[w][c] = output_temp[w];
									    }
                }
                for(int w = 0; w < iter; w++)	////iter ?? (Tw-2)/stride
      					{				
                #pragma HLS pipeline II = 1													
  									out[((h)*WK+ w + WB*TW_2_STRIDE)*CTC + ct/Tc] = output_buf0[w];
      					}
    						
    					}
					else	
					{
						for(int c = 0; c < Tc; c++) //pipeline
						{	
							#pragma HLS pipeline II = 1
							
							if(h != H_OUT)
							{
							  	output_temp = out_stream.read();
							}
						
								for(int w = 0; w < 14; w++)	//unroll
								{
                                                    
									output_buf0[w][c] = output_temp[w];
								}
            }
            for(int w = 0; w < TW_2_STRIDE; w++)	//unroll
					  {				
            #pragma HLS pipeline II = 1						
										out[((h)*WK+ w + wb*TW_2_STRIDE)*CTC + ct/Tc] = output_buf0[w];									
		      	}															
								
					}
      			}
     		}
  	}        
}
//for(int w = 0; w < iter ; w++)		
        					
                   
        					
extern "C" {
void depthwise(
	      hls::vector<DTYPE, Tw>* in0,    
          hls::vector<DTYPE, 9>* kernel,
          hls::vector<DTYPE, Tc>* out,
          int W,
          int H,
          int C,
          int K,
          int pad,
          int stride,
          int num_nonzero,
          int iter
          )
{

#pragma HLS INTERFACE m_axi port = in0 bundle = gmem0 num_read_outstanding=32 num_write_outstanding=32
#pragma HLS INTERFACE m_axi port = out bundle = gmem1 num_read_outstanding=32 num_write_outstanding=32
#pragma HLS INTERFACE m_axi port = kernel bundle = gmem1
static hls::stream<hls::vector<DTYPE, Tw> > in0_stream("in0_stream"); 

static hls::stream<hls::vector<DTYPE, (Tw-2)> > out0_stream("out0_stream");

//output
#pragma HLS dataflow
load_input(in0, in0_stream, W,H,C,K, pad, stride, num_nonzero);//in0_stream,

depthwise_conv(in0_stream, kernel,out0_stream, W,H,C,K, pad, stride, num_nonzero, iter);
store_output( out0_stream, out, W,H,C,K, pad, stride, num_nonzero, iter);

} // end function

}