# frozen_string_literal: true

require "tensorrt_rb/tensorrt_rb"

module TensorRT
  VERSION = "0.1.0"

  autoload :CUDA, 'tensorrt/cuda'
end
