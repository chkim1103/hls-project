# hls-project

Depthwise는 3개의 layer에 맞춰서 돌리면 됩니다. Depthwise.cpp 파일을 사용하면 됩니다. 

Depthwise.h에서 파라미터 숫자를 
Conv dw / s1 3 × 3 × 32 dw 112 × 112 × 32         
Conv dw / s1 3 × 3 × 256 dw 28 × 28 × 256       
Conv dw / s2 3 × 3 × 1024 dw 7 × 7 × 1024
에 맞춰서 make run 하면 됩니다. 

Pointwise 또한 3개의 layer에 맞춰서 돌리면 됩니다.

Pointwise.h에서 파라미터 숫자를
Conv / s1 1 × 1 × 32 × 64 112 × 112 × 32
Conv / s1 1 × 1 × 128 × 256 28 × 28 × 128
Conv / s1 1 × 1 × 1024 × 1024 7 × 7 × 1024
에 맞춰서 make run 하면 됩니다. 





Baseline
gcc pointwise_openblas.c -o pointwise_openblas -I /usr/include/openblas -lopenblas -g 
./pointwise_openblas
gcc depthwise_openblas.c -o depthwise_openblas -I /usr/include/openblas -lopenblas -g 
./depthwise_openblas
