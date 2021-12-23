#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"

#include <iostream>
#include "depthwise.h"
#define Tc 16

//VER1. H ?? PIPELINE??? ??. channel? 16?? ?? -> reg ?? ?? version

void load_input(       hls::vector<DTYPE, 16>* in,
                       hls::stream<hls::vector<DTYPE, 16> >& inStream,
                       char W,
                       char H,
                       char C,
                       char k,
                       char pad,
                       char stride
                       ) 
{
  DTYPE* in_dtype = (DTYPE*)in;
  hls::vector<DTYPE, 16>* in_temp;
  hls::vector<DTYPE, 16> temp_row;
  char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? Tw-2 : ((W-K+2*pad)+1)%(Tw-2);
  char num_nonzero = temp + 2 - pad;
  bool mask_nonzero[Tw];
  
  for(int i = 0; i < Tw; i++)
  {
      mask_nonzero[i] = ((num_nonzero-i)>0) ? 1:0; 
  }
	for(int ct = 0; ct < C; ct += Tc)
	{
		for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2) + 1; wb++)    
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
  					in_temp = (hls::vector<DTYPE, 16>*)(in_dtype + ((c + ct)*W*H + h*W + wb*(Tw-2) - pad));
  					temp_row = *in_temp;
  					temp_row[0] = (pad && (wb==0)) ? 0 : temp_row[0];
  					
  					if(wb == ((W-K+2*pad)+1-1)/(Tw-2))
  					{
  						for(int i = 0; i < 16; i++)
  						{
  							temp_row[i] = (mask_nonzero[i]) ?  temp_row[i] : 0;
  						}
  					}
  					inStream.write(temp_row);
	 	  	}
   	} 
             
				for(int c = 0; c < Tc; c += 1)
				{
        #pragma HLS pipeline II=1
  				temp_row = 0;
  				inStream.write(temp_row);
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
// void test(       
//                  hls::stream<hls::vector<DTYPE, 16> >& inStream,
//                  hls::vector<DTYPE, 16>* out,
//                  char W,
//                  char H,
//                  char C,
//                  char k,
//                  char pad,
//                  char stride)
// {
//     for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2) + 1; wb++) 
//     {
//         #pragma HLS pipeline II=1
//         for(int h = 0; h < H + 2*pad ; h++)
//         {
// //            #pragma HLS pipeline II=1
//             hls::vector<DTYPE, 16> out_temp = inStream.read();
        
//             out[0] = out_temp;
//         }
//     }
// }

// void partition_input( hls::stream<hls::vector<DTYPE, 16> >& inStream,
//                       hls::stream<hls::vector<DTYPE, 3> >& inStream_partition,
//                       char W,
//                       char H,
//                       char C,
//                       char k,
//                       char pad,
//                       char stride)
// {   
//     for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2); wb++)
//     {
//         for(int h = 0; h < H + 2*pad; h++) 
//         {   
//         #pragma HLS pipeline II = 1
//             hls::vector<DTYPE, 16> temp_row = inStream.read();
//             for(int w = 0; w < (Tw-2)/stride ; w++)
//             {
// //             #pragma HLS pipeline II = 1
//                 hls::vector<DTYPE, 3> temp;
//                 temp[0] = temp_row[w*stride];
//                 temp[1] = temp_row[w*stride + 1];
//                 temp[2] = temp_row[w*stride + 2];
//                 inStream_partition.write(temp);
//             }
  
//         }
//     }
    
//     // for wb = ((W-K+2*pad)+1-1)/(Tw-2)
//     char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? Tw-2 : ((W-K+2*pad)+1)%(Tw-2);
//     char num_nonzero = temp + 2 -pad;
//     char iter = ((temp+stride-1)/stride <= 1) ? 1 : (temp+stride-1)/stride;
    
//     for(int h = 0; h < H + 2*pad; h++) 
//         {
//         #pragma HLS pipeline II = 1
//             hls::vector<DTYPE, 16> temp_row = inStream.read();
//             for(int w = 0; w < iter; w++)
//             {

//                 hls::vector<DTYPE, 3> temp;
//                 temp[0] =  temp_row[w*stride];
//                 temp[1] =  temp_row[w*stride + 1];
//                 temp[2] =  temp_row[w*stride + 2];
//                 inStream_partition.write(temp);
//             }
  
//         }
      
// }

void depthwise_conv(hls::stream<hls::vector<DTYPE, 16> >& in0_stream,
                    //hls::stream<hls::vector<DTYPE, 16> >& in1_stream,
                    //hls::stream<hls::vector<DTYPE, 16> >& in2_stream,                     
                    //hls::stream<hls::vector<DTYPE, 16> >& in3_stream,
                    DTYPE* kernel,                                        
                    hls::stream<hls::vector<DTYPE, 16> >&out0_stream, 
				//	hls::stream<hls::vector<DTYPE, 4> >&out1_stream, 
				//	hls::stream<hls::vector<DTYPE, 4> >&out2_stream, 
				//	hls::stream<hls::vector<DTYPE, 4> >&out3_stream, 
                    char W,
                    char H,
                    char C,
                    char k,
                    char pad,
                    char stride
                    )
{
    static DTYPE kernel_buf[16][3][3];
    static DTYPE output_buf[16][2][Tw-2];
    static DTYPE output_temp[16][2][Tw-2];
    hls::vector<DTYPE, 16> store_output0[(Tw-2)];
	  hls::vector<DTYPE, 16> store_output1[(Tw-2)];
    hls::vector<DTYPE, (Tc)> test0[Tw-2];
    hls::vector<DTYPE, (Tc)> test1[Tw-2];
    for(int i = 0 ; i < Tw-2; i++)
    {
        test0[i] = 10;
        test1[i] = 11;
    }
    
    char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? Tw-2 : ((W-K+2*pad)+1)%(Tw-2);
    char num_nonzero = temp + 2-pad;
    char iter = ((temp+stride-1)/stride <= 1) ? 1 : (temp+stride-1)/stride;
	
	#pragma HLS ARRAY_PARTITION variable=kernel_buf complete dim=0
	#pragma HLS ARRAY_PARTITION variable=output_buf complete dim=0
	#pragma HLS ARRAY_PARTITION variable=output_temp complete dim=0  
	#pragma HLS ARRAY_PARTITION variable=store_output0 complete dim=0 
	#pragma HLS ARRAY_PARTITION variable=store_output1 complete dim=0 
	
    for(int i = 0; i < 4; i++)
    {
        //#pragma HLS UNROLL		
		#pragma HLS pipeline II = 1
        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < K; k++)
            {
			//#pragma HLS LOOP_TRIPCOUNT min = 3 max = 3
                kernel_buf[i][j][k] = kernel[i*K*K + K*j + k];
            }
        }
    }
   
  for(int ct = 0; ct < C; ct += Tc)
	{  
		for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2)+1; wb++)
		{	
			if(wb == ((W-K+2*pad)+1-1)/(Tw-2))
			{
				for(int h = 0; h < H+ 2*pad+1 ; h++)     
				{   
					for(int c = 0; c < Tc ; c++)
					{  
						#pragma HLS pipeline II = 1     
						hls::vector<DTYPE, 16> input_temp0 = in0_stream.read();	

						if(h%2==0)
						{
							for(int w = 0; w < (iter) ; w++)  //pipeline 
							{
							//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
								output_buf[c][0][w] = input_temp0[stride*w]*kernel_buf[c][0][0]+input_temp0[stride*w+1]*kernel_buf[c][0][1]+input_temp0[stride*w+2]*kernel_buf[c][0][2];
								output_buf[c][1][w] = input_temp0[stride*w]*kernel_buf[c][1][0]+input_temp0[stride*w+1]*kernel_buf[c][1][1]+input_temp0[stride*w+2]*kernel_buf[c][1][2]+output_temp[c][0][w];
								store_output0[w][c] = input_temp0[stride*w]*kernel_buf[c][2][0]+input_temp0[stride*w+1]*kernel_buf[c][2][1]+input_temp0[stride*w+2]*kernel_buf[c][2][2]+output_temp[c][1][w];
														
								output_temp[c][0][w] = output_buf[c][0][w];
								output_temp[c][1][w] = output_buf[c][1][w];					
							}
							if(h>2&&((h - 1)%stride==0)&&(c < iter))
							{   								
								out0_stream.write(store_output1[c]);	
							}
						}											
						else
						{
							for(int w = 0; w < (iter) ; w++)  //pipeline 
							{
							//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
								output_buf[c][0][w] = input_temp0[stride*w]*kernel_buf[c][0][0]+input_temp0[stride*w+1]*kernel_buf[c][0][1]+input_temp0[stride*w+2]*kernel_buf[c][0][2];
								output_buf[c][1][w] = input_temp0[stride*w]*kernel_buf[c][1][0]+input_temp0[stride*w+1]*kernel_buf[c][1][1]+input_temp0[stride*w+2]*kernel_buf[c][1][2]+output_temp[c][0][w];
								store_output1[w][c] = input_temp0[stride*w]*kernel_buf[c][2][0]+input_temp0[stride*w+1]*kernel_buf[c][2][1]+input_temp0[stride*w+2]*kernel_buf[c][2][2]+output_temp[c][1][w];
														
								output_temp[c][0][w] = output_buf[c][0][w];
								output_temp[c][1][w] = output_buf[c][1][w];					
							}
							if(h>2&&((h - 1)%stride==0)&&(c < iter))
							{   								
								out0_stream.write(store_output0[c]);	
							}
						}	
					}	
				}
			}
			//#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1    
			else
			{
				for(int h = 0; h < H+ 2*pad + 1 ; h++)     
				{   
					for(int c = 0; c < Tc ; c++)
					{  
						#pragma HLS pipeline II = 1     
						hls::vector<DTYPE, 16> input_temp0 = in0_stream.read();												
						if(h%2==0)
						{
							for(int w = 0; w < (Tw-2)/stride ; w++)  //pipeline 
							{
                                                
							//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
								output_buf[c][0][w] = input_temp0[stride*w]*kernel_buf[c][0][0]+input_temp0[stride*w+1]*kernel_buf[c][0][1]+input_temp0[stride*w+2]*kernel_buf[c][0][2];
								output_buf[c][1][w] = input_temp0[stride*w]*kernel_buf[c][1][0]+input_temp0[stride*w+1]*kernel_buf[c][1][1]+input_temp0[stride*w+2]*kernel_buf[c][1][2]+output_temp[c][0][w];
								store_output0[w][c] = input_temp0[stride*w]*kernel_buf[c][2][0]+input_temp0[stride*w+1]*kernel_buf[c][2][1]+input_temp0[stride*w+2]*kernel_buf[c][2][2]+output_temp[c][1][w];
														
								output_temp[c][0][w] = output_buf[c][0][w];
								output_temp[c][1][w] = output_buf[c][1][w];					
							}
							if(h>2&&((h - 1)%stride==0)&&(c < (Tw-2)/stride))
							{   								
								out0_stream.write(store_output1[c]);	
							}
						}											
						else
						{
							for(int w = 0; w < (Tw-2)/stride ; w++)  //pipeline 
							{
							//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
								output_buf[c][0][w] = input_temp0[stride*w]*kernel_buf[c][0][0]+input_temp0[stride*w+1]*kernel_buf[c][0][1]+input_temp0[stride*w+2]*kernel_buf[c][0][2];
								output_buf[c][1][w] = input_temp0[stride*w]*kernel_buf[c][1][0]+input_temp0[stride*w+1]*kernel_buf[c][1][1]+input_temp0[stride*w+2]*kernel_buf[c][1][2]+output_temp[c][0][w];
								store_output1[w][c] = input_temp0[stride*w]*kernel_buf[c][2][0]+input_temp0[stride*w+1]*kernel_buf[c][2][1]+input_temp0[stride*w+2]*kernel_buf[c][2][2]+output_temp[c][1][w];
														
								output_temp[c][0][w] = output_buf[c][0][w];
								output_temp[c][1][w] = output_buf[c][1][w];					
							}
							if(h>2&&((h - 1)%stride==0)&&(c < (Tw-2)/stride))
							{   								
								out0_stream.write(store_output0[c]);	
							}
						}
					}							
				}
			}
		}
	}
    // for wb = ((W-K+2*pad)+1-1)/(Tw-2)
      
}

void store_output(    hls::stream<hls::vector<DTYPE, Tc> >& out_stream,				  
                      hls::vector<DTYPE, Tc>* out,
                      char W,
                      char H,
                      char C,
                      char k,
                      char pad,
                      char stride	)
{  
	char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? (Tw-2) : ((W-K+2*pad)+1)%(Tw-2);
  char num_nonzero = temp + 2-pad;    
  char iter = ((temp+stride-1)/stride <= 1) ? 1 : (temp+stride-1)/stride;	// ????? w ??

hls::vector<DTYPE, (Tc)> test0[Tw-2];
hls::vector<DTYPE, (Tc)> test1[Tw-2];
    for(int i = 0 ; i < Tw-2; i++)
    {
      for(int j = 0; j < 16; j++)
      {
        test0[i][j] = 10;
        test1[i][j] = 11;
      }
    }
    
  	for(int ct = 0; ct < C; ct += Tc)
  	{
    		for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2) + 1; wb++) 
    		{
      			for(int h = 0; h < ((H-K+ 2*pad)/stride+1); h++) 
      			{		
        				if(wb == ((W-K+2*pad)+1-1)/(Tw-2))	
        				{ 
        					for(int w = 0; w < iter ; w++)		
        					{	
                   #pragma HLS pipeline II = 1
        					out[(h*((W-K+2*pad)/stride+1)+ w + ((W-K+2*pad)+1-1)/(Tw-2)*(Tw-2)/stride)*C/Tc + ct/Tc] = out_stream.read();
        					}
        				}
        				else	
        				{
        					for(int w = 0; w < (Tw/2)/stride ; w++)		
        					{	
                   #pragma HLS pipeline II = 1
        					out[(h*((W-K+2*pad)/stride+1)+ w + wb*(Tw-2)/stride)*C/Tc + ct/Tc] = out_stream.read();                               
        					}				
        				}
      			}
     		}
  	}        
}

extern "C" {
void depthwise(
	      hls::vector<DTYPE, 16>* in0,
          //hls::vector<DTYPE, 16>* in1,
          //hls::vector<DTYPE, 16>* in2,
          //hls::vector<DTYPE, 16>* in3,       
          DTYPE* kernel,
          hls::vector<DTYPE, Tc>* out0,
		  //hls::vector<DTYPE, 16>* out1,
		  //hls::vector<DTYPE, 16>* out2,
		  //hls::vector<DTYPE, 16>* out3,
          int W,
          int H,
          int C,
          int K,
          int pad,
          int stride
          )
{

#pragma HLS INTERFACE m_axi port = in0 bundle = gmem0 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = in1 bundle = gmem1 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = in2 bundle = gmem2 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = in3 bundle = gmem3 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = kernel bundle = gmem3 //max_read_burst_length = 2 max_write_burst_length = 2
#pragma HLS INTERFACE m_axi port = out0 bundle = gmem0 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = out1 bundle = gmem1 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = out2 bundle = gmem2 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = out3 bundle = gmem3 //max_read_burst_length = 2 max_write_burst_length = 2

static hls::stream<hls::vector<DTYPE, 16> > in0_stream("in0_stream"); 
//static hls::stream<hls::vector<DTYPE, 16> > in1_stream("in1_stream");
//static hls::stream<hls::vector<DTYPE, 16> > in2_stream("in2_stream");
//static hls::stream<hls::vector<DTYPE, 16> > in3_stream("in3_stream");

//static hls::stream<hls::vector<DTYPE, 3> > in0_stream_partition("in0_stream_partition");
//static hls::stream<hls::vector<DTYPE, 3> > in1_stream_partition("in1_stream_partition");
//static hls::stream<hls::vector<DTYPE, 3> > in2_stream_partition("in2_stream_partition");
//static hls::stream<hls::vector<DTYPE, 3> > in3_stream_partition("in3_stream_partition");

static hls::stream<hls::vector<DTYPE, Tc> > out0_stream("out0_stream");
//static hls::stream<hls::vector<DTYPE, 4> > out1_stream("out1_stream");
//static hls::stream<hls::vector<DTYPE, 4> > out2_stream("out2_stream");
//static hls::stream<hls::vector<DTYPE, 4> > out3_stream("out3_stream");

char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? (Tw-2) : ((W-K+2*pad)+1)%(Tw-2);
char num_nonzero = temp + 2-pad;    
char iter = ((temp+stride-1)/stride <= 1) ? 1 : (temp+stride-1)/stride;

//char mode_out0 = (stride==2) ? 4 : 2;
//char mode_out1 = (stride==2) ? 4 : 2;
//char mode_out2 = (stride==2) ? 3 : 2;
//char mode_out3 = (stride==2) ? 3 : 1;

//char iter_w0 = (iter+3)/4;
//char iter_w1 = (iter+2)/4;
//char iter_w2 = (iter+1)/4;
//char iter_w3 = (iter)/4;


//output
#pragma HLS dataflow
load_input(in0, in0_stream, W,H,C,K, pad, stride);//in0_stream,
//test(in0_stream, out, W,H,C,K, pad, stride);
//load_input(in1, in1_stream, W,H,C,K, pad, stride);
//load_input(in2, in2_stream, W,H,C,K, pad, stride);
//load_input(in3, in3_stream, W,H,C,K, pad, stride);

//partition_input(in0_stream, in0_stream_partition, W,H,C,K, pad, stride);
//partition_input(in1_stream, in1_stream_partition, W,H,C,K, pad, stride);
//partition_input(in2_stream, in2_stream_partition, W,H,C,K, pad, stride);
//partition_input(in3_stream, in3_stream_partition, W,H,C,K, pad, stride);

depthwise_conv(in0_stream, kernel,out0_stream, W,H,C,K, pad, stride);
store_output( out0_stream, out0, W,H,C,K, pad, stride);
//store_output( out1_stream, out1, W,H,C,K, pad, stride, mode_out1, iter_w1);
//store_output( out2_stream, out2, W,H,C,K, pad, stride, mode_out2, iter_w2);
//store_output( out3_stream, out3, W,H,C,K, pad, stride, mode_out3, iter_w3);

} // end function

}