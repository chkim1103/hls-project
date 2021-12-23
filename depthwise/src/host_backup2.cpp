/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
#include <time.h>
#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include "depthwise.h"
#include <math.h>
const int SIZE = 512;
void print_tensor(DTYPE* output, int w, int h, int c)
{
//    printf("%s\n", name);
    for(int k = 0; k < c; k++)
    {
      for(int j = 0; j < h; j ++)
      {
          for(int i = 0; i < w ; i++)
          {
              printf(" %lf", output[k*w*h + j*w + i]);
          }
          printf("\n");
      }
      printf("\n");
      printf("\n");
    }
}

void print_mat(DTYPE* output, int w, int h)
{
//    printf("%s\n", name);
      for(int j = 0; j < h; j ++)
      {
          for(int i = 0; i < w ; i++)
          {
              printf(" %lf", output[j*w + i]);
          }
          printf("\n");
      }
    
}
void transpose(DTYPE *input, DTYPE *output, int W, int H, int C)
{
  for(int c = 0; c < C; c++) {
  	for(int h = 0; h < H; h++) {
  		for(int w = 0; w < W; w++){   
            output[h*C*W + w*C + c] = input[c*W*H + h*W + w];
      }   
    }
  }                
}

void depthwise_sw(DTYPE *input,  DTYPE *kernel, DTYPE *output, int W, int H, int C) {
  for(int c = 0; c < C; c++) {
  	for(int h = 0; h < (H-K+1); h++) {
  		for(int w = 0; w < (W-K+1); w++){
  					output[c*(W-K+1)*(H-K+1) + h*(W-K+1) + w] = 0;
                	for (int ki = 0; ki < K; ki++) {
                		for (int kj = 0; kj < K; kj++) {
							output[c*(W-K+1)*(H-K+1) + h*(W-K+1) + w] += kernel[K*K*c + ki*K + kj] * input[W*H*c + W*(h+ki) + w + kj];
					}
				}
			}
		}
  } 
}

//depthwise_sw_2(input.data(), kernel.data(), out_temp.data(), W+2*pad, H+2*pad, C);
void depthwise_sw_2(DTYPE *input,  DTYPE *kernel, DTYPE *output, int W, int H, int C) {
  for(int c = 0; c < C; c++) {
  	for(int h = 0; h < ((H-K)/STRIDE+1); h++) {
  		for(int w = 0; w < ((W-K)/STRIDE+1); w++){
  					output[c*((W-K)/STRIDE+1)*((H-K)/STRIDE+1) + h*((W-K)/STRIDE+1) + w] = 0;
                	for (int ki = 0; ki < K; ki++) {
                		for (int kj = 0; kj < K; kj++) {
							output[c*((W-K)/STRIDE+1)*((H-K)/STRIDE+1) + h*((W-K)/STRIDE+1) + w] += kernel[K*K*c + ki*K + kj] * input[W*H*c + W*(h*STRIDE+ki) + w*STRIDE+ kj];
					}
				}
			}
		}
  } 
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    int pad = PAD;
    int stride = STRIDE;
    std::string binaryFile = argv[1];
    size_t input_vector_size_bytes = sizeof(float) * W*H*C;
    size_t kernel_vector_size_bytes = sizeof(float) * K*K*C;
    size_t output_vector_size_bytes = sizeof(float) * ((W-K+2*pad)/stride+1)*((H-K+2*pad)/stride+1)*C;
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_depthwise;
    cl::CommandQueue q;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
    // hood user ptr
    // is used if it is properly aligned. when not aligned, runtime had no choice
    // but to create
    // its own host side buffer. So it is recommended to use this allocator if
    // user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
    // boundary. It will
    // ensure that user buffer is used when user create Buffer/Mem object with
    // CL_MEM_USE_HOST_PTR
    //DTYPE* input0; 
    //DTYPE* input1;
    //DTYPE* input2; 
    //DTYPE* input3;
    std::vector<float, aligned_allocator<float> > input(C*(W+2*pad)*(H+2*pad));
    
    std::vector<float, aligned_allocator<float> > input0(W*H);
    std::vector<float, aligned_allocator<float> > input1(W*H);
    std::vector<float, aligned_allocator<float> > input2(W*H);
    std::vector<float, aligned_allocator<float> > input3(W*H);
    
    //DTYPE input0[W*H];
    //DTYPE input1[W*H];
    //DTYPE input2[W*H];
    //DTYPE input3[W*H];
    
    //std::vector<float, aligned_allocator<float> > input0(W*H);
    //std::vector<float, aligned_allocator<float> > input1(W*H);
    //std::vector<float, aligned_allocator<float> > input2(W*H);
    //std::vector<float, aligned_allocator<float> > input3(W*H);
    std::vector<float, aligned_allocator<float> > kernel(C*K*K);
    
    std::vector<float, aligned_allocator<float> > out_hw(((W-K+2*pad)/stride+1)*((H-K+2*pad)/stride+1)*C); //+2*pad

    // Create the test data
    for(int c = 0; c < C;c++)
    {
        for (int i = 0; i < K*K; i++) {
            kernel[c*K*K + i] = i;
        }
    }    
    for(int c = 0; c < C; c++) 
    {
        for(int h=0; h<H+2*pad; h++)
        {
            for(int w=0; w<W+2*pad; w++)
            {   
            
                if(((w==0)&&pad) || ((h==0)&&pad) || ((w==W+2*pad-1)&&pad) || ((h==H+2*pad-1)&&pad))
                {
                   input[c*(W+2*pad)*(H+2*pad) + h*(W+2*pad) + w] = 0;   
                } 
                else
                {
                    input[c*(W+2*pad)*(H+2*pad) + h*(W+2*pad) + w] = (c*W*H + (h-pad)*W + (w-pad))%20;
                    switch(c)
                    {
                        case 0: input0[(h-pad)*(W) + w-pad] = input[c*(W+2*pad)*(H+2*pad) + h*(W+2*pad) + w];
                                break;
                        case 1: input1[(h-pad)*(W) + w-pad] = input[c*(W+2*pad)*(H+2*pad) + h*(W+2*pad) + w];
                                break;
                        case 2: input2[(h-pad)*(W) + w-pad] = input[c*(W+2*pad)*(H+2*pad) + h*(W+2*pad) + w];
                                break;
                        case 3: input3[(h-pad)*(W) + w-pad] = input[c*(W+2*pad)*(H+2*pad) + h*(W+2*pad) + w];
                                break;
                    }
                }
                       
            }
              
        }    
    }
    print_tensor(input.data(),W+2*pad,H+2*pad,C);
//    print_mat(input0, W, H);
//    print_mat(input1, W, H);
//    print_mat(input2, W, H);
//    print_mat(input3, W, H);
    
    for (int i = 0; i < ((W-K+2*pad)/stride+1)*((H-K+ 2*pad)/stride+1)*C; i++) {//+2*pad
        out_hw[i] = -1;
    }
    
    
    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
 
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_depthwise = cl::Kernel(program, "depthwise", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
 
    for(int c = 0; c < C/4; c++)
    {
          //input0 = input.data() + W*H*c;
          //input1 = input.data() + W*H*(c+1);
          //input2 = input.data() + W*H*(c+2);
          //input3 = input.data() + W*H*(c+3);
    
          OCL_CHECK(err, cl::Buffer buffer_in0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_vector_size_bytes,
                                               input0.data(), &err));
          OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_vector_size_bytes,
                                               input1.data(), &err));
          OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_vector_size_bytes,
                                               input2.data(), &err));
          OCL_CHECK(err, cl::Buffer buffer_in3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_vector_size_bytes,
                                                input3.data(), &err));         
          OCL_CHECK(err, cl::Buffer buffer_kernel(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, kernel_vector_size_bytes,
                                               kernel.data(), &err));                                                                                            
          OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, output_vector_size_bytes,
                                                out_hw.data(), &err));
          //OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, test_vector_size_bytes,
          //                                       test.data(), &err));
                                              
      
          OCL_CHECK(err, err = krnl_depthwise.setArg(0, buffer_in0));
          OCL_CHECK(err, err = krnl_depthwise.setArg(1, buffer_in1));
          OCL_CHECK(err, err = krnl_depthwise.setArg(2, buffer_in2));
          OCL_CHECK(err, err = krnl_depthwise.setArg(3, buffer_in3));
          OCL_CHECK(err, err = krnl_depthwise.setArg(4, buffer_kernel));
          OCL_CHECK(err, err = krnl_depthwise.setArg(5, buffer_output));
          OCL_CHECK(err, err = krnl_depthwise.setArg(6, W));
          OCL_CHECK(err, err = krnl_depthwise.setArg(7, H));
          OCL_CHECK(err, err = krnl_depthwise.setArg(8, C));
          OCL_CHECK(err, err = krnl_depthwise.setArg(9, K));
          OCL_CHECK(err, err = krnl_depthwise.setArg(10, pad));
          OCL_CHECK(err, err = krnl_depthwise.setArg(11, stride));
          // Copy input data to device global memory
         OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in0, buffer_in1, buffer_in2, buffer_in3,buffer_kernel}, 0 ));/* 0 means from host*/
          // Launch the Kernel
          // For HLS kernels global and local size is always (1,1,1). So, it is
          // recommended
          // to always use enqueueTask() for invoking HLS kernel
          //auto start = std::chrono::steady_clock::now();
          OCL_CHECK(err, err = q.enqueueTask(krnl_depthwise));
          //auto end = std::chrono::steady_clock::now();
        	//std::cout << "Done.\n";
        	//double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        	//double gops = double(SIZE) * SIZE* SIZE * SIZE * 2 / (exec_time);  ", GOPS: " << gops
        	//std::cout << "Time: " << exec_time*1e-9 <<  std::endl;
      
          // Copy Result from Device Global Memory to Host Local Memory
          OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
      
          q.finish();
    }
    
    //print_mat(test.data(),(16),(3));
  
    // OPENCL HOST CODE AREA END
    
    printf("hw\n");
    print_mat(out_hw.data(), C, ((H-K+2*pad)/stride+1)*((W-K+2*pad)/stride+1)  );//+2*pad
    // Compare the results of the Device to the simulation
    printf("end!\n");
    return (0);
}