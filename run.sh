
# # clear
# rm main

# # rm compile.log
# annoy_build_flag=${1}_BUILD

# nvcc -o main main.cu -D${annoy_build_flag} #2> compile.log
# # code compile.log

# # rm run.log
# ./main #> run.log
# # code run.log

#----------------------------------


# nvcc -o shfl_test shfl_test.cu 
# ./shfl_test 

#----------------------------------


# cd ~/fnlPrj/annoy/src
# rm ./annoy/annoylib.so
# # rm compile.log
# nvcc --shared -o annoylib.so annoymodule.cu --compiler-options '-fPIC' -I/usr/include/python3.6 -DANNOYLIB_GPU_BUILD #2> compile.log
# # code compile.log
# mv annoylib.so ./annoy

python3 main.py

