# frozen_string_literal: true

require "mkmf-rice"

# TensorRT/CUDA paths for Linux (x86_64 and aarch64)
case RUBY_PLATFORM
when /x86_64-linux/
  tensorrt_include = ENV["TENSORRT_INCLUDE"] || "/usr/include/x86_64-linux-gnu"
  tensorrt_lib = ENV["TENSORRT_LIB"] || "/usr/lib/x86_64-linux-gnu"
  cuda_include = ENV["CUDA_INCLUDE"] || "/usr/local/cuda/include"
  cuda_lib = ENV["CUDA_LIB"] || "/usr/local/cuda/lib64"
when /aarch64-linux/
  tensorrt_include = ENV["TENSORRT_INCLUDE"] || "/usr/include/aarch64-linux-gnu"
  tensorrt_lib = ENV["TENSORRT_LIB"] || "/usr/lib/aarch64-linux-gnu"
  cuda_include = ENV["CUDA_INCLUDE"] || "/usr/local/cuda/include"
  cuda_lib = ENV["CUDA_LIB"] || "/usr/local/cuda/lib64"
else
  abort "Unsupported platform: #{RUBY_PLATFORM}. Only Linux x86_64 and aarch64 are supported."
end

$INCFLAGS << " -I#{tensorrt_include} -I#{cuda_include}"
$LDFLAGS << " -L#{tensorrt_lib} -L#{cuda_lib}"
$LDFLAGS << " -lnvinfer -lcudart"

# C++17 for Rice
$CXXFLAGS << " -std=c++17"

create_makefile("tensorrt_rb/tensorrt_rb")
