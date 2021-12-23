#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"

#include <iostream>
#include "depthwise.h"

//VER1. H ?? PIPELINE??? ??. channel? ?? ?? -> reg ?? ?? version

void load_input(       hls::vector<DTYPE, Tw>* in,
					   hls::vector<DTYPE, 9>* kernel,
                       hls::stream<hls::vector<DTYPE, Tw> >& inStream0,
					   hls::stream<hls::vector<DTYPE, 9> >& kerStream0,

					   hls::stream<hls::vector<DTYPE, Tw> >& inStream1,
					   hls::stream<hls::vector<DTYPE, 9> >& kerStream1,

					   hls::stream<hls::vector<DTYPE, Tw> >& inStream2,
					   hls::stream<hls::vector<DTYPE, 9> >& kerStream2,

					   hls::stream<hls::vector<DTYPE, Tw> >& inStream3,
					   hls::stream<hls::vector<DTYPE, 9> >& kerStream3,
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
  hls::vector<DTYPE, 9> kernel_1ch = 0;
 	#pragma HLS ARRAY_PARTITION variable=mask_nonzero complete dim=0 
  
  static int WB_1 = ((W-K+2*pad)+1-1)/(Tw-2) + 1;
  static int WB = ((W-K+2*pad)+1-1)/(Tw-2);
  static int WH = W*H;
  static int TW_2 = Tw-2;
  static bool mode = 0;
  
  for(int i = 0; i < Tw; i++)
  {  
      #pragma HLS UNROLL
      mask_nonzero[i] = ((num_nonzero-i)>0) ? 1:0; 
  }


	for(int ct = 0; ct < C; ct += Tc)
	{	
			kernel_1ch = kernel[ct];
			kerStream0.write(kernel_1ch);

			kernel_1ch = kernel[ct + 1];
			kerStream1.write(kernel_1ch);

			kernel_1ch = kernel[ct + 2];
			kerStream2.write(kernel_1ch);

			kernel_1ch = kernel[ct + 3];
			kerStream3.write(kernel_1ch);
				

		for(int wb = 0; wb < WB; wb++)    
		{
			mode = 1;
			if(pad)
			{
				//채널 unroll

				temp_row = 0;
				inStream0.write(temp_row);
				inStream1.write(temp_row);
				inStream2.write(temp_row);
				inStream3.write(temp_row);				
			}
			
			for(int h = 0; h < H; h++) 
			{
				//채널 unroll
					#pragma HLS pipeline II=1
					hls::vector<DTYPE, Tw>* in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((ct)*WH + h*W + wb*(TW_2) - pad));
					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
					temp_row[0] = (pad && (wb==0)) ? 0 : temp_row[0];
					inStream0.write(temp_row);

					in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((1 + ct)*WH + h*W + wb*(TW_2) - pad));
					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
					temp_row[0] = (pad && (wb==0)) ? 0 : temp_row[0];
					inStream1.write(temp_row);

					in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((2 + ct)*WH + h*W + wb*(TW_2) - pad));
					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
					temp_row[0] = (pad && (wb==0)) ? 0 : temp_row[0];
					inStream2.write(temp_row);

					in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((3 + ct)*WH + h*W + wb*(TW_2) - pad));
					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
					temp_row[0] = (pad && (wb==0)) ? 0 : temp_row[0];
					inStream3.write(temp_row);
				
			} 
		
			if(pad)
			{
				//채널 unroll
					temp_row = 0;
					inStream0.write(temp_row);
					inStream1.write(temp_row);
					inStream2.write(temp_row);
					inStream3.write(temp_row);
				
			}
				
		}
			if(pad)
			{
				//채널 unroll
					temp_row = 0;
					inStream0.write(temp_row);
					inStream1.write(temp_row);
					inStream2.write(temp_row);
					inStream3.write(temp_row);
				
			}
			
			for(int h = 0; h < H; h++) 
			{
 
  					#pragma HLS pipeline II=1
  					hls::vector<DTYPE, Tw>* in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((ct)*WH + h*W + WB*(TW_2) - pad));
  					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
  					temp_row[0] = (pad && (mode==0)) ? 0 : temp_row[0];
  					
					for(int i = 0; i < Tw; i++)
					{
						temp_row[i] = (mask_nonzero[i]) ?  temp_row[i] : 0;
					}
  					inStream0.write(temp_row);

					in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((1 + ct)*WH + h*W + WB*(TW_2) - pad));
  					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
  					temp_row[0] = (pad && (mode==0)) ? 0 : temp_row[0];
  					
					for(int i = 0; i < Tw; i++)
					{
						temp_row[i] = (mask_nonzero[i]) ?  temp_row[i] : 0;
					}
  					inStream1.write(temp_row);

					in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((2 + ct)*WH + h*W + WB*(TW_2) - pad));
  					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
  					temp_row[0] = (pad && (mode==0)) ? 0 : temp_row[0];
  					
					for(int i = 0; i < Tw; i++)
					{
						temp_row[i] = (mask_nonzero[i]) ?  temp_row[i] : 0;
					}
  					inStream2.write(temp_row);

					in_temp = (hls::vector<DTYPE, Tw>*)(in_dtype + ((3 + ct)*WH + h*W + WB*(TW_2) - pad));
  					temp_row = *in_temp; //*(in+((c + ct)*W*H + h*W + wb*(Tw-2) - pad)/16);
  					temp_row[0] = (pad && (mode==0)) ? 0 : temp_row[0];
  					
					for(int i = 0; i < Tw; i++)
					{
						temp_row[i] = (mask_nonzero[i]) ?  temp_row[i] : 0;
					}
  					inStream3.write(temp_row);
	 	  		
   			} 
			
			if(pad)
			{				
				#pragma HLS pipeline II=1
					temp_row = 0;
					inStream0.write(temp_row);
					inStream1.write(temp_row);
					inStream2.write(temp_row);
					inStream3.write(temp_row);				
			}
				
		
	}
}





void depthwise_conv_1ch(hls::stream<hls::vector<DTYPE, Tw> >& in_stream,
                    hls::stream<hls::vector<DTYPE, 9> >& ker_stream,                                        
                    hls::stream<hls::vector<DTYPE, (Tw-2)> >&out_stream, 
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
    DTYPE output_buf[2][Tw-2] = {0};
    DTYPE output_temp[2][Tw-2]= {0};
    hls::vector<DTYPE, 9> kernel = 0;
	hls::vector<DTYPE, (Tw-2)> store_output;

	#pragma HLS ARRAY_PARTITION variable=output_buf complete dim=0
	#pragma HLS ARRAY_PARTITION variable=output_temp complete dim=0  
	
 int WB_1 = ((W-K+2*pad)+1-1)/(Tw-2) + 1;
 int WB   = ((W-K+2*pad)+1-1)/(Tw-2);
 int H_2PAD_2 = H + 2*pad - 2;;
 
if(stride == 1)
{
	for(int ct = 0; ct < C; ct += Tc)
	{  
		kernel = ker_stream.read();

		for(int wb = 0; wb < WB_1 ; wb++)
		{		
				for(int i = 0; i < 2; i++)
				{
					#pragma HLS PIPELINE II=1
					hls::vector<DTYPE, Tw> input_temp = in_stream.read();						
					for(int w = 0; w < 14 ; w++)  //???? ITER (iter)
					{
					//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
						output_buf[0][w] = input_temp[stride*w]*kernel[0]+input_temp[stride*w+1]*kernel[1]+input_temp[stride*w+2]*kernel[2];
						output_buf[1][w] = input_temp[stride*w]*kernel[3]+input_temp[stride*w+1]*kernel[4]+input_temp[stride*w+2]*kernel[5]+output_temp[0][w];
						store_output[w] = input_temp[stride*w]*kernel[6]+input_temp[stride*w+1]*kernel[7]+input_temp[stride*w+2]*kernel[8]+output_temp[1][w];
												
						output_temp[0][w] = output_buf[0][w];
						output_temp[1][w] = output_buf[1][w];					
					}	
				}	
				for(int h = 0; h < H_2PAD_2 ; h++)     
				{   	
					#pragma HLS pipeline II = 1 			    
					hls::vector<DTYPE, Tw> input_temp = in_stream.read();						
					for(int w = 0; w < 14 ; w++)  //???? ITER (iter)
					{						
					//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
						output_buf[0][w] = input_temp[stride*w]*kernel[0]+input_temp[stride*w+1]*kernel[1]+input_temp[stride*w+2]*kernel[2];
						output_buf[1][w] = input_temp[stride*w]*kernel[3]+input_temp[stride*w+1]*kernel[4]+input_temp[stride*w+2]*kernel[5]+output_temp[0][w];
						store_output[w] = input_temp[stride*w]*kernel[6]+input_temp[stride*w+1]*kernel[7]+input_temp[stride*w+2]*kernel[8]+output_temp[1][w];
												
						output_temp[0][w] = output_buf[0][w];
						output_temp[1][w] = output_buf[1][w];					
					}			
					//이 위치에서 store		
					out_stream.write(store_output);
				}				
		
								
			//#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1    
		}
	}
}

else
{
	for(int ct = 0; ct < C; ct += Tc)
	{  
		kernel = ker_stream.read();

		for(int wb = 0; wb < WB_1 ; wb++)
		{		
				
			hls::vector<DTYPE, Tw> input_temp = in_stream.read();						
			for(int w = 0; w < 14 ; w++)  //???? ITER (iter)
			{
			//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
				output_buf[0][w] = input_temp[stride*w]*kernel[0]+input_temp[stride*w+1]*kernel[1]+input_temp[stride*w+2]*kernel[2];
				output_buf[1][w] = input_temp[stride*w]*kernel[3]+input_temp[stride*w+1]*kernel[4]+input_temp[stride*w+2]*kernel[5]+output_temp[0][w];
				store_output[w] = input_temp[stride*w]*kernel[6]+input_temp[stride*w+1]*kernel[7]+input_temp[stride*w+2]*kernel[8]+output_temp[1][w];
										
				output_temp[0][w] = output_buf[0][w];
				output_temp[1][w] = output_buf[1][w];					
			}	
					
			for(int h = 0; h < H_2PAD_2 ; h+=2)     
			{   
				for(int i = 0; i < 2; i++)
				{
					#pragma HLS pipeline II = 1 			    
					hls::vector<DTYPE, Tw> input_temp = in_stream.read();						
					for(int w = 0; w < 14 ; w++)  //???? ITER (iter)
					{						
					//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
						output_buf[0][w] = input_temp[stride*w]*kernel[0]+input_temp[stride*w+1]*kernel[1]+input_temp[stride*w+2]*kernel[2];
						output_buf[1][w] = input_temp[stride*w]*kernel[3]+input_temp[stride*w+1]*kernel[4]+input_temp[stride*w+2]*kernel[5]+output_temp[0][w];
						store_output[w] = input_temp[stride*w]*kernel[6]+input_temp[stride*w+1]*kernel[7]+input_temp[stride*w+2]*kernel[8]+output_temp[1][w];
												
						output_temp[0][w] = output_buf[0][w];
						output_temp[1][w] = output_buf[1][w];					
					}
				}			
				//이 위치에서 store		
				out_stream.write(store_output);
			}				
			//#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1    
		}
	}
}
    // for wb = WB
      
}


void store_output(    	hls::stream<hls::vector<DTYPE, (Tw-2)> >& out_stream0,
						hls::stream<hls::vector<DTYPE, (Tw-2)> >& out_stream1,
						hls::stream<hls::vector<DTYPE, (Tw-2)> >& out_stream2,
						hls::stream<hls::vector<DTYPE, (Tw-2)> >& out_stream3,				  
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
 int iteration = TW_2_STRIDE;

static hls::vector<DTYPE, Tw-2> output_temp0 = 0;
static hls::vector<DTYPE, Tw-2> output_temp1 = 0;
static hls::vector<DTYPE, Tw-2> output_temp2 = 0;
static hls::vector<DTYPE, Tw-2> output_temp3 = 0;
static hls::vector<DTYPE, Tc> output_buf0[Tw-2] = {0};


static hls::vector<DTYPE, Tc> test = 0;

	for(int ct = 0; ct < C; ct += Tc)
	{
		iteration = TW_2_STRIDE;

		for(int wb = 0; wb < WB; wb++) 
		{
			for(int h = 0; h < H_OUT; h++) 
			{	
				#pragma HLS pipeline II = 1							    
					output_temp0 = out_stream0.read();			
					for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
					{
						output_buf0[w][0] = output_temp0[w];
					}

					output_temp1 = out_stream1.read();			
					for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
					{
						output_buf0[w][1] = output_temp1[w];
					}

					output_temp2 = out_stream2.read();			
					for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
					{
						output_buf0[w][2] = output_temp2[w];
					}

					output_temp3 = out_stream3.read();			
					for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
					{
						output_buf0[w][3] = output_temp3[w];
					}
				
				for(int w = 0; w < iteration; w++)	////iter ?? (Tw-2)/stride
				{				
					#pragma HLS pipeline II = 1													
					out[((h)*WK+ w + wb*TW_2_STRIDE)*CTC + ct/Tc] = output_buf0[w];
				}						
			}
		}

		iteration = iter;

		for(int h = 0; h < H_OUT; h++) 
		{			
				#pragma HLS pipeline II = 1
				output_temp0 = out_stream0.read();			
				for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
				{
					output_buf0[w][0] = output_temp0[w];
				}

				output_temp1 = out_stream1.read();			
				for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
				{
					output_buf0[w][1] = output_temp1[w];
				}

				output_temp2 = out_stream2.read();			
				for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
				{
					output_buf0[w][2] = output_temp2[w];
				}

				output_temp3 = out_stream3.read();			
				for(int w = 0; w < 14; w++)	////iter ?? (Tw-2)/stride
				{
					output_buf0[w][3] = output_temp3[w];
				}
		
			for(int w = 0; w < iteration; w++)	//unroll
			{				
				#pragma HLS pipeline II = 1						
				out[((h)*WK+ w + WB*TW_2_STRIDE)*CTC + ct/Tc] = output_buf0[w];									
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
static hls::stream<hls::vector<DTYPE, Tw> > in_stream0("in_stream0"); 
static hls::stream<hls::vector<DTYPE, Tw> > in_stream1("in_stream1"); 
static hls::stream<hls::vector<DTYPE, Tw> > in_stream2("in_stream2"); 
static hls::stream<hls::vector<DTYPE, Tw> > in_stream3("in_stream3"); 

static hls::stream<hls::vector<DTYPE, 9> > ker_stream0("ker_stream0"); 
static hls::stream<hls::vector<DTYPE, 9> > ker_stream1("ker_stream0"); 
static hls::stream<hls::vector<DTYPE, 9> > ker_stream2("ker_stream0"); 
static hls::stream<hls::vector<DTYPE, 9> > ker_stream3("ker_stream0"); 

static hls::stream<hls::vector<DTYPE, (Tw-2)> > out_stream0("out_stream0");
static hls::stream<hls::vector<DTYPE, (Tw-2)> > out_stream1("out_stream1");
static hls::stream<hls::vector<DTYPE, (Tw-2)> > out_stream2("out_stream2");
static hls::stream<hls::vector<DTYPE, (Tw-2)> > out_stream3("out_stream3");



//output
#pragma HLS dataflow
load_input(in0,kernel, in_stream0, ker_stream0, in_stream1, ker_stream1, in_stream2, ker_stream2,
in_stream3, ker_stream3, W,H,C,K, pad, stride, num_nonzero);//in0_stream,

depthwise_conv_1ch(in_stream0, ker_stream0, out_stream0, W,H,C,K, pad, stride, num_nonzero, iter);
depthwise_conv_1ch(in_stream1, ker_stream1, out_stream1, W,H,C,K, pad, stride, num_nonzero, iter);
depthwise_conv_1ch(in_stream2, ker_stream2, out_stream2, W,H,C,K, pad, stride, num_nonzero, iter);
depthwise_conv_1ch(in_stream3, ker_stream3, out_stream3, W,H,C,K, pad, stride, num_nonzero, iter);

store_output( out_stream0,out_stream1,out_stream2,out_stream3, out, W,H,C,K, pad, stride, num_nonzero, iter);

} // end function

}