#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"

#include <iostream>
#include "depthwise.h"



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

 for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2) + 1; wb++)    //16 , 15일때 모두 0이어야함. 17일때 1 이어야함.
  {
      if(pad)
      {
          temp_row = 0;
          inStream.write(temp_row);
      }
    
     for(int h = 0; h < H; h++) 
      {
          #pragma HLS pipeline II=1
          in_temp = (hls::vector<DTYPE, 16>*)(in_dtype + (h*W + wb*(Tw-2) - pad));
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
          if(pad)
          {   
              temp_row = 0;
              inStream.write(temp_row);
          }
    
  }
}
void test(       
                 hls::stream<hls::vector<DTYPE, 16> >& inStream,
                 hls::vector<DTYPE, 16>* out,
                 char W,
                 char H,
                 char C,
                 char k,
                 char pad,
                 char stride)
{
    for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2) + 1; wb++) 
    {
        #pragma HLS pipeline II=1
        for(int h = 0; h < H + 2*pad ; h++)
        {
//            #pragma HLS pipeline II=1
            hls::vector<DTYPE, 16> out_temp = inStream.read();
        
            out[0] = out_temp;
        }
    }
}

void partition_input( hls::stream<hls::vector<DTYPE, 16> >& inStream,
                      hls::stream<hls::vector<DTYPE, 3> >& inStream_partition,
                      char W,
                      char H,
                      char C,
                      char k,
                      char pad,
                      char stride)
{   
    for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2); wb++)
    {
        for(int h = 0; h < H + 2*pad; h++) 
        {   
        #pragma HLS pipeline II = 1
            hls::vector<DTYPE, 16> temp_row = inStream.read();
            for(int w = 0; w < (Tw-2)/stride ; w++)
            {
//             #pragma HLS pipeline II = 1
                hls::vector<DTYPE, 3> temp;
                temp[0] = temp_row[w*stride];
                temp[1] = temp_row[w*stride + 1];
                temp[2] = temp_row[w*stride + 2];
                inStream_partition.write(temp);
            }
  
        }
    }
    
    // for wb = ((W-K+2*pad)+1-1)/(Tw-2)
    char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? Tw-2 : ((W-K+2*pad)+1)%(Tw-2);
    char num_nonzero = temp + 2 -pad;
    char iter = ((temp+stride-1)/stride <= 1) ? 1 : (temp+stride-1)/stride;
    
    for(int h = 0; h < H + 2*pad; h++) 
        {
        #pragma HLS pipeline II = 1
            hls::vector<DTYPE, 16> temp_row = inStream.read();
            for(int w = 0; w < iter; w++)
            {

                hls::vector<DTYPE, 3> temp;
                temp[0] =  temp_row[w*stride];
                temp[1] =  temp_row[w*stride + 1];
                temp[2] =  temp_row[w*stride + 2];
                inStream_partition.write(temp);
            }
  
        }
      
}

void depthwise_conv(hls::stream<hls::vector<DTYPE, 3> >& in0_stream,
                    hls::stream<hls::vector<DTYPE, 3> >& in1_stream,
                    hls::stream<hls::vector<DTYPE, 3> >& in2_stream,                     
                    hls::stream<hls::vector<DTYPE, 3> >& in3_stream,
                    DTYPE* kernel,                                        
                    hls::stream<hls::vector<DTYPE, 4> >&out_stream, 
                    char W,
                    char H,
                    char C,
                    char k,
                    char pad,
                    char stride
                    )
{
    hls::vector<DTYPE, 3> input_temp0;
    hls::vector<DTYPE, 3> input_temp1;
    hls::vector<DTYPE, 3> input_temp2;
    hls::vector<DTYPE, 3> input_temp3;
    static DTYPE kernel_buf[4][3][3];
    static DTYPE output_buf[4][3][Tw-2];
    static DTYPE output_temp[4][2][Tw-2];
    
    hls::vector<DTYPE, 4> store_output;
    
    char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? Tw-2 : ((W-K+2*pad)+1)%(Tw-2);
    char num_nonzero = temp + 2-pad;
    char iter = ((temp+stride-1)/stride <= 1) ? 1 : (temp+stride-1)/stride;
  #pragma HLS ARRAY_PARTITION variable=kernel_buf complete dim=0
  #pragma HLS ARRAY_PARTITION variable=output_buf complete dim=0
  #pragma HLS ARRAY_PARTITION variable=output_temp complete dim=0  
    
    for(int i = 0; i < 4; i++)
    {
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
   
      
    for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2); wb++)
    {
        //#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1    
        for(int h = 0; h < H+ 2*pad ; h++)     
        {     
            #pragma HLS pipeline II = 1        
            for(int w = 0; w < (Tw-2)/stride ; w++)  //pipeline 
            {
            //#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
                        
                hls::vector<DTYPE, 3> input_temp0 = in0_stream.read();
                hls::vector<DTYPE, 3> input_temp1 = in1_stream.read();
                hls::vector<DTYPE, 3> input_temp2 = in2_stream.read();
                hls::vector<DTYPE, 3> input_temp3 = in3_stream.read();
                
                output_buf[0][0][w] = input_temp0[0]*kernel_buf[0][0][0]+input_temp0[1]*kernel_buf[0][0][1]+input_temp0[2]*kernel_buf[0][0][2];
                output_buf[0][1][w] = input_temp0[0]*kernel_buf[0][1][0]+input_temp0[1]*kernel_buf[0][1][1]+input_temp0[2]*kernel_buf[0][1][2]+output_temp[0][0][w];
                output_buf[0][2][w] = input_temp0[0]*kernel_buf[0][2][0]+input_temp0[1]*kernel_buf[0][2][1]+input_temp0[2]*kernel_buf[0][2][2]+output_temp[0][1][w];

                output_buf[1][0][w] = input_temp1[0]*kernel_buf[1][0][0]+input_temp1[1]*kernel_buf[1][0][1]+input_temp1[2]*kernel_buf[1][0][2];
                output_buf[1][1][w] = input_temp1[0]*kernel_buf[1][1][0]+input_temp1[1]*kernel_buf[1][1][1]+input_temp1[2]*kernel_buf[1][1][2]+output_temp[1][0][w];
                output_buf[1][2][w] = input_temp1[0]*kernel_buf[1][2][0]+input_temp1[1]*kernel_buf[1][2][1]+input_temp1[2]*kernel_buf[1][2][2]+output_temp[1][1][w];

                output_buf[2][0][w] = input_temp2[0]*kernel_buf[2][0][0]+input_temp2[1]*kernel_buf[2][0][1]+input_temp2[2]*kernel_buf[2][0][2];
                output_buf[2][1][w] = input_temp2[0]*kernel_buf[2][1][0]+input_temp2[1]*kernel_buf[2][1][1]+input_temp2[2]*kernel_buf[2][1][2]+output_temp[2][0][w];
                output_buf[2][2][w] = input_temp2[0]*kernel_buf[2][2][0]+input_temp2[1]*kernel_buf[2][2][1]+input_temp2[2]*kernel_buf[2][2][2]+output_temp[2][1][w];

                output_buf[3][0][w] = input_temp3[0]*kernel_buf[3][0][0]+input_temp3[1]*kernel_buf[3][0][1]+input_temp3[2]*kernel_buf[3][0][2];
                output_buf[3][1][w] = input_temp3[0]*kernel_buf[3][1][0]+input_temp3[1]*kernel_buf[3][1][1]+input_temp3[2]*kernel_buf[3][1][2]+output_temp[3][0][w];
                output_buf[3][2][w] = input_temp3[0]*kernel_buf[3][2][0]+input_temp3[1]*kernel_buf[3][2][1]+input_temp3[2]*kernel_buf[3][2][2]+output_temp[3][1][w];
                
                if(h>1&&(h%stride==0))
                {
                    for(int i = 0; i < 4; i++)
                    {            
                        store_output[i] = output_buf[i][2][w];
                    }         
                    //out[h*(W-k+1) + w] = store_output;
                    out_stream.write(store_output);
                }
                
            }
            for(int w = 0; w < (Tw-2)/stride ; w++)
            {
            	#pragma HLS UNROLL
		 	      	//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
            	output_temp[0][0][w] = output_buf[0][0][w];
            	output_temp[0][1][w] = output_buf[0][1][w];
            	output_temp[1][0][w] = output_buf[1][0][w];
            	output_temp[1][1][w] = output_buf[1][1][w];
            	output_temp[2][0][w] = output_buf[2][0][w];
      				output_temp[2][1][w] = output_buf[2][1][w];
      				output_temp[3][0][w] = output_buf[3][0][w];
      				output_temp[3][1][w] = output_buf[3][1][w];

            }
            //store at here
        }
    }
    
    // for wb = ((W-K+2*pad)+1-1)/(Tw-2)
        for(int h = 0; h < H+ 2*pad ; h++)     
        {     
            #pragma HLS pipeline II = 1        
            for(int w = 0; w < (iter) ; w++)  //pipeline 
            {
            //#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
            //#pragma HLS pipeline II = 1            
                hls::vector<DTYPE, 3> input_temp0 = in0_stream.read();
                hls::vector<DTYPE, 3> input_temp1 = in1_stream.read();
                hls::vector<DTYPE, 3> input_temp2 = in2_stream.read();
                hls::vector<DTYPE, 3> input_temp3 = in3_stream.read();
                
                output_buf[0][0][w] = input_temp0[0]*kernel_buf[0][0][0]+input_temp0[1]*kernel_buf[0][0][1]+input_temp0[2]*kernel_buf[0][0][2];
                output_buf[0][1][w] = input_temp0[0]*kernel_buf[0][1][0]+input_temp0[1]*kernel_buf[0][1][1]+input_temp0[2]*kernel_buf[0][1][2]+output_temp[0][0][w];
                output_buf[0][2][w] = input_temp0[0]*kernel_buf[0][2][0]+input_temp0[1]*kernel_buf[0][2][1]+input_temp0[2]*kernel_buf[0][2][2]+output_temp[0][1][w];

                output_buf[1][0][w] = input_temp1[0]*kernel_buf[1][0][0]+input_temp1[1]*kernel_buf[1][0][1]+input_temp1[2]*kernel_buf[1][0][2];
                output_buf[1][1][w] = input_temp1[0]*kernel_buf[1][1][0]+input_temp1[1]*kernel_buf[1][1][1]+input_temp1[2]*kernel_buf[1][1][2]+output_temp[1][0][w];
                output_buf[1][2][w] = input_temp1[0]*kernel_buf[1][2][0]+input_temp1[1]*kernel_buf[1][2][1]+input_temp1[2]*kernel_buf[1][2][2]+output_temp[1][1][w];

                output_buf[2][0][w] = input_temp2[0]*kernel_buf[2][0][0]+input_temp2[1]*kernel_buf[2][0][1]+input_temp2[2]*kernel_buf[2][0][2];
                output_buf[2][1][w] = input_temp2[0]*kernel_buf[2][1][0]+input_temp2[1]*kernel_buf[2][1][1]+input_temp2[2]*kernel_buf[2][1][2]+output_temp[2][0][w];
                output_buf[2][2][w] = input_temp2[0]*kernel_buf[2][2][0]+input_temp2[1]*kernel_buf[2][2][1]+input_temp2[2]*kernel_buf[2][2][2]+output_temp[2][1][w];

                output_buf[3][0][w] = input_temp3[0]*kernel_buf[3][0][0]+input_temp3[1]*kernel_buf[3][0][1]+input_temp3[2]*kernel_buf[3][0][2];
                output_buf[3][1][w] = input_temp3[0]*kernel_buf[3][1][0]+input_temp3[1]*kernel_buf[3][1][1]+input_temp3[2]*kernel_buf[3][1][2]+output_temp[3][0][w];
                output_buf[3][2][w] = input_temp3[0]*kernel_buf[3][2][0]+input_temp3[1]*kernel_buf[3][2][1]+input_temp3[2]*kernel_buf[3][2][2]+output_temp[3][1][w];
                if(h>1&&(h%stride == 0 ))
                {
                    for(int i = 0; i < 4; i++)
                    {            
                        store_output[i] = output_buf[i][2][w];
                    }         
                    //out[h*(W-k+1) + w] = store_output;
                    out_stream.write(store_output);
                }
                
            }
            for(int w = 0; w < (iter) ; w++)
            {
            	#pragma HLS UNROLL
		 	      	//#pragma HLS LOOP_TRIPCOUNT min = 14 max = 14
            	output_temp[0][0][w] = output_buf[0][0][w];
            	output_temp[0][1][w] = output_buf[0][1][w];
            	output_temp[1][0][w] = output_buf[1][0][w];
            	output_temp[1][1][w] = output_buf[1][1][w];
            	output_temp[2][0][w] = output_buf[2][0][w];
      				output_temp[2][1][w] = output_buf[2][1][w];
      				output_temp[3][0][w] = output_buf[3][0][w];
      				output_temp[3][1][w] = output_buf[3][1][w];

            }
            //store at here
        }
}

void store_output( hls::stream<hls::vector<DTYPE, 4> >& outStream,
                      hls::vector<DTYPE, 4>* out,
                      char W,
                      char H,
                      char C,
                      char k,
                      char pad,
                      char stride)
{  
   for(int wb = 0; wb < ((W-K+2*pad)+1-1)/(Tw-2); wb++) 
   {
      for(int h = 0; h < ((H-K+ 2*pad)/stride+1); h++) 
      {
          #pragma HLS pipeline II = 1
          for(int w = 0; w < (Tw-2)/stride ; w++)
          {
//            #pragma HLS pipeline II = 1
            out[((h*((W-K+2*pad)/stride+1))+ w + wb*(Tw-2)/stride)] = outStream.read();
          }

      }
   }
    char temp = (((W-K+2*pad)+1)%(Tw-2) == 0) ? (Tw-2) : ((W-K+2*pad)+1)%(Tw-2);
    char num_nonzero = temp + 2-pad;    
    char iter = ((temp+stride-1)/stride <= 1) ? 1 : (temp+stride-1)/stride;
    // for wb = ((W-K+2*pad)+1-1)/(Tw-2)
       for(int h = 0; h < ((H-K+ 2*pad)/stride+1); h++) 
      {
          #pragma HLS pipeline II = 1
          for(int w = 0; w < iter ; w++)
          {
//            
            out[((h*((W-K+2*pad)/stride+1))+ w + (((W-K+2*pad)+1-1)/(Tw-2))*(Tw-2)/stride)] = outStream.read();
          }

      }
}

extern "C" {
void depthwise(
	        hls::vector<DTYPE, 16>* in0,
          hls::vector<DTYPE, 16>* in1,
          hls::vector<DTYPE, 16>* in2,
          hls::vector<DTYPE, 16>* in3,       
          DTYPE* kernel,
          hls::vector<DTYPE, 4>* out,
          int W,
          int H,
          int C,
          int K,
          int pad,
          int stride
          )
{

#pragma HLS INTERFACE m_axi port = in0 bundle = gmem0 //max_read_burst_length = 2 max_write_burst_length = 2
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem1 //max_read_burst_length = 2 max_write_burst_length = 2
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem2 //max_read_burst_length = 2 max_write_burst_length = 2
#pragma HLS INTERFACE m_axi port = in3 bundle = gmem3 //max_read_burst_length = 2 max_write_burst_length = 2
//#pragma HLS INTERFACE m_axi port = kernel bundle = gmem3 //max_read_burst_length = 2 max_write_burst_length = 2
#pragma HLS INTERFACE m_axi port = out bundle = gmem0 //max_read_burst_length = 2 max_write_burst_length = 2

static hls::stream<hls::vector<DTYPE, 16> > in0_stream("in0_stream"); 
static hls::stream<hls::vector<DTYPE, 16> > in1_stream("in1_stream");
static hls::stream<hls::vector<DTYPE, 16> > in2_stream("in2_stream");
static hls::stream<hls::vector<DTYPE, 16> > in3_stream("in3_stream");

static hls::stream<hls::vector<DTYPE, 3> > in0_stream_partition("in0_stream_partition");
static hls::stream<hls::vector<DTYPE, 3> > in1_stream_partition("in1_stream_partition");
static hls::stream<hls::vector<DTYPE, 3> > in2_stream_partition("in2_stream_partition");
static hls::stream<hls::vector<DTYPE, 3> > in3_stream_partition("in3_stream_partition");

static hls::stream<hls::vector<DTYPE, 4> > out_stream("out_stream");
 


//output
#pragma HLS dataflow
load_input(in0, in0_stream, W,H,C,K, pad, stride);//in0_stream,
//test(in0_stream, out, W,H,C,K, pad, stride);
load_input(in1, in1_stream, W,H,C,K, pad, stride);
load_input(in2, in2_stream, W,H,C,K, pad, stride);
load_input(in3, in3_stream, W,H,C,K, pad, stride);

partition_input(in0_stream, in0_stream_partition, W,H,C,K, pad, stride);
partition_input(in1_stream, in1_stream_partition, W,H,C,K, pad, stride);
partition_input(in2_stream, in2_stream_partition, W,H,C,K, pad, stride);
partition_input(in3_stream, in3_stream_partition, W,H,C,K, pad, stride);

depthwise_conv(in0_stream_partition, in1_stream_partition, in2_stream_partition, in3_stream_partition, kernel,out_stream, W,H,C,K, pad, stride);
store_output( out_stream, out, W,H,C,K, pad, stride);

} // end function

}