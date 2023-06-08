# clear
# rm main
# g++ -o main main.cc
# ./main


#----------------------------------

clear
rm main
# rm compile.log
nvcc -o main main.cu #2> compile.log
# code compile.log

rm run.log
./main > run.log
code run.log