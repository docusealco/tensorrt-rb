# frozen_string_literal: true

require "ffi"

module TensorRT
  module CUDA
    extend FFI::Library

    CUDA_LIBS = %w[
      libcudart.so
      libcudart.so.12
      libcudart.so.11
    ].freeze

    begin
      ffi_lib CUDA_LIBS
    rescue LoadError => e
      raise LoadError, "Could not load CUDA runtime library. Tried: #{CUDA_LIBS.join(', ')}. Error: #{e.message}"
    end

    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2

    attach_function :cudaMalloc, [:pointer, :size_t], :int
    attach_function :cudaFree, [:pointer], :int
    attach_function :cudaMemcpy, [:pointer, :pointer, :size_t, :int], :int
    attach_function :cudaMemcpyAsync, [:pointer, :pointer, :size_t, :int, :pointer], :int
    attach_function :cudaDeviceSynchronize, [], :int
    attach_function :cudaStreamSynchronize, [:pointer], :int

    Error = Class.new(StandardError)

    class << self
      def malloc(size)
        ptr = FFI::MemoryPointer.new(:pointer)
        err = cudaMalloc(ptr, size)

        raise Error, "cudaMalloc failed with error #{err}" unless err.zero?

        ptr.read_pointer.address
      end

      def free(ptr)
        err = cudaFree(FFI::Pointer.new(ptr))

        raise Error, "cudaFree failed with error #{err}" unless err.zero?
      end

      def memcpy_htod(dst, src_ptr, size)
        err = cudaMemcpy(FFI::Pointer.new(dst), src_ptr, size, MEMCPY_HOST_TO_DEVICE)

        raise Error, "cudaMemcpy H2D failed with error #{err}" unless err.zero?
      end

      def memcpy_dtoh(dst_ptr, src, size)
        err = cudaMemcpy(dst_ptr, FFI::Pointer.new(src), size, MEMCPY_DEVICE_TO_HOST)

        raise Error, "cudaMemcpy D2H failed with error #{err}" unless err.zero?
      end

      def memcpy_htod_async(dst, src_ptr, size, stream)
        err = cudaMemcpyAsync(FFI::Pointer.new(dst), src_ptr, size, MEMCPY_HOST_TO_DEVICE, FFI::Pointer.new(stream))

        raise Error, "cudaMemcpyAsync H2D failed with error #{err}" unless err.zero?
      end

      def memcpy_dtoh_async(dst_ptr, src, size, stream)
        err = cudaMemcpyAsync(dst_ptr, FFI::Pointer.new(src), size, MEMCPY_DEVICE_TO_HOST, FFI::Pointer.new(stream))

        raise Error, "cudaMemcpyAsync D2H failed with error #{err}" unless err.zero?
      end

      def synchronize
        err = cudaDeviceSynchronize

        raise Error, "cudaDeviceSynchronize failed with error #{err}" unless err.zero?
      end

      def stream_synchronize(stream)
        err = cudaStreamSynchronize(FFI::Pointer.new(stream))

        raise Error, "cudaStreamSynchronize failed with error #{err}" unless err.zero?
      end
    end

    freeze
  end
end
