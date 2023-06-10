
# clear
rm main
# rm AnnoyGPU.tree
# rm compile.log

annoy_build_flag=${1}_BUILD

nvcc -o main main.cu -D${annoy_build_flag} #2> compile.log
# code compile.log

# rm run.log
./main #> run.log
# code run.log

#----------------------------------


# nvcc -o shfl_test shfl_test.cu 
# ./shfl_test 