# tensorrt-rb

Minimal TensorRT bindings for Ruby using Rice (C++ bindings).

## Requirements

- Linux (x86_64 or aarch64)
- Ruby >= 3.0
- TensorRT (with headers and libraries)
- CUDA runtime
- Rice gem (`gem install rice`)

## Installation

```bash
cd tensorrt-rb

# Set paths if not in standard locations
export TENSORRT_INCLUDE=/path/to/tensorrt/include
export TENSORRT_LIB=/path/to/tensorrt/lib
export CUDA_INCLUDE=/usr/local/cuda/include
export CUDA_LIB=/usr/local/cuda/lib64

# Build
rake compile

# Or install as gem
gem build tensorrt.gemspec
gem install tensorrt-*.gem
```

### Default Library Paths

**x86_64:**
- TensorRT: `/usr/include/x86_64-linux-gnu`, `/usr/lib/x86_64-linux-gnu`
- CUDA: `/usr/local/cuda/include`, `/usr/local/cuda/lib64`

**aarch64:**
- TensorRT: `/usr/include/aarch64-linux-gnu`, `/usr/lib/aarch64-linux-gnu`
- CUDA: `/usr/local/cuda/include`, `/usr/local/cuda/lib64`

## API

### TensorRT::Engine

```ruby
engine = TensorRT::Engine.new(path, verbose: false)

# Tensor info
engine.num_io_tensors                  # Number of input/output tensors
engine.get_tensor_name(index)          # Tensor name by index
engine.is_input?(name)                 # Check if tensor is input
engine.get_tensor_shape(name)          # Shape as array [1, 3, 640, 640]
engine.get_tensor_bytes(name)          # Size in bytes

# Memory binding
engine.set_tensor_address(name, device_ptr)

# Inference
engine.execute                         # Synchronous (blocking)
engine.enqueue                         # Asynchronous (non-blocking)

# Stream management
engine.get_stream                      # CUDA stream handle (uint64)
engine.stream_synchronize              # Wait for stream completion
```

### TensorRT::CUDA

```ruby
# Memory allocation
ptr = TensorRT::CUDA.malloc(bytes)
TensorRT::CUDA.free(ptr)

# Synchronous copy
TensorRT::CUDA.memcpy_htod(device_ptr, host_ptr, bytes)  # Host → Device
TensorRT::CUDA.memcpy_dtoh(host_ptr, device_ptr, bytes)  # Device → Host

# Asynchronous copy
TensorRT::CUDA.memcpy_htod_async(device_ptr, host_ptr, bytes, stream)
TensorRT::CUDA.memcpy_dtoh_async(host_ptr, device_ptr, bytes, stream)

# Synchronization
TensorRT::CUDA.synchronize                    # All operations
TensorRT::CUDA.stream_synchronize(stream)     # Specific stream
```

## Examples

### Synchronous Inference

```ruby
require "tensorrt"

engine = TensorRT::Engine.new("model.engine")

# Allocate GPU memory
input_bytes = engine.get_tensor_bytes("input")
output_bytes = engine.get_tensor_bytes("output")
output_size = engine.get_tensor_shape("output").reduce(1, :*)

input_device = TensorRT::CUDA.malloc(input_bytes)
output_device = TensorRT::CUDA.malloc(output_bytes)
engine.set_tensor_address("input", input_device)
engine.set_tensor_address("output", output_device)

# Prepare input data
input_data = preprocess_image(image_path)  # Returns Numo::SFloat
input_host = FFI::MemoryPointer.new(:float, input_data.size)
input_host.write_bytes(input_data.to_binary)

# Copy input to GPU
TensorRT::CUDA.memcpy_htod(input_device, input_host, input_bytes)

# Run inference
engine.execute

# Copy output from GPU
output_host = FFI::MemoryPointer.new(:float, output_size)
TensorRT::CUDA.memcpy_dtoh(output_host, output_device, output_bytes)
output_data = output_host.read_array_of_float(output_size)

# Cleanup
TensorRT::CUDA.free(input_device)
TensorRT::CUDA.free(output_device)
```

### Pipelined Async Inference

Overlap CPU preprocessing with GPU inference for maximum throughput:

```ruby
require "tensorrt"

engine = TensorRT::Engine.new("model.engine")
stream = engine.get_stream

# Allocate GPU memory
input_bytes = engine.get_tensor_bytes("input")
output_bytes = engine.get_tensor_bytes("output")
output_size = engine.get_tensor_shape("output").reduce(1, :*)

input_device = TensorRT::CUDA.malloc(input_bytes)
output_device = TensorRT::CUDA.malloc(output_bytes)
engine.set_tensor_address("input", input_device)
engine.set_tensor_address("output", output_device)

# Allocate host buffers
input_host = FFI::MemoryPointer.new(:float, input_bytes / 4)
output_host = FFI::MemoryPointer.new(:float, output_size)

# Preload first image
current_image = preprocess_image(images[0])

images.each_with_index do |image_path, i|
  # Copy current image to GPU (async)
  input_host.write_bytes(current_image.to_binary)
  TensorRT::CUDA.memcpy_htod_async(input_device, input_host, input_bytes, stream)

  # Start async inference
  engine.enqueue

  # Preprocess next image on CPU while GPU is busy
  next_image = preprocess_image(images[i + 1]) if i < images.size - 1

  # Wait for GPU inference to complete
  engine.stream_synchronize

  # Copy output from GPU
  TensorRT::CUDA.memcpy_dtoh(output_host, output_device, output_bytes)
  output_data = output_host.read_array_of_float(output_size)

  # Process results
  process_detections(output_data)

  current_image = next_image
end

# Cleanup
TensorRT::CUDA.free(input_device)
TensorRT::CUDA.free(output_device)
```
