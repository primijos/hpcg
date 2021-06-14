#export CROSS_COMPILE=arm-linux-gnueabihf-
unset CROSS_COMPILE

#export WHAT=bitstream
export WHAT=bin/xhpcg

#export MODE=-DOMPSS_ONLY_SMP
unset MODE

export PATH=$PATH:~/Programs/Xilinx/Vivado/2020.1/bin/

make arch=OmpSs_at_fpga_MPI MODE=$MODE $WHAT

# bitstream
#PATH=$PATH:~/Programs/Xilinx/Vivado/2020.1/bin/ make arch=OmpSs_at_fpga CXXFLAGS=-DOMPSS_ONLY_SMP bitstream
#PATH=$PATH:~/Programs/Xilinx/Vivado/2020.1/bin/ make arch=OmpSs_at_fpga bitstream
#PATH=$PATH:~/Programs/Xilinx/Vivado/2020.1/bin/ make arch=OmpSs_at_fpga bin/xhpcg
#PATH=$PATH:~/Programs/Xilinx/Vivado/2020.1/bin/ make arch=OmpSs_at_fpga MODE=-DOMPSS_ONLY_SMP bin/xhpcg
