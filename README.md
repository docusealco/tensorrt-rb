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
gem build tensorrt-rb.gemspec
gem install tensorrt-rb-*.gem
```

### Default Library Paths

**x86_64:**
- TensorRT: `/usr/include/x86_64-linux-gnu`, `/usr/lib/x86_64-linux-gnu`
- CUDA: `/usr/local/cuda/include`, `/usr/local/cuda/lib64`

**aarch64:**
- TensorRT: `/usr/include/aarch64-linux-gnu`, `/usr/lib/aarch64-linux-gnu`
- CUDA: `/usr/local/cuda/include`, `/usr/local/cuda/lib64`

## API

```ruby
require "tensorrt"

# Load engine
engine = TensorRT::Engine.new("model.engine", verbose: false)

# Tensor information methods
engine.num_io_tensors           # Number of input/output tensors
engine.get_tensor_name(0)       # Get tensor name by index
engine.is_input?("input")       # Check if tensor is input
engine.get_tensor_shape("input") # Get tensor shape [1, 3, 640, 640]
engine.get_tensor_bytes("input") # Get tensor size in bytes

# Synchronous inference
engine.set_tensor_address("input", device_ptr)   # Set tensor memory address
engine.set_tensor_address("output", device_ptr)  # Set output tensor address
engine.execute                  # Execute inference (blocking)

# Asynchronous inference
engine.enqueue                  # Queue inference on stream (non-blocking)
engine.stream_synchronize       # Wait for stream to complete
engine.get_stream               # Get CUDA stream handle (uint64)
```

### Async Inference with CUDA Streams

The engine includes a built-in CUDA stream for asynchronous operations:

```ruby
# Async inference (non-blocking)
engine.enqueue                 # Queue inference on stream
engine.stream_synchronize      # Wait for stream to complete

# Get stream handle (for use with external CUDA operations)
stream_ptr = engine.get_stream # Returns uint64 stream address
```

### Pipelined Inference Example

Use async inference to overlap GPU compute with CPU preprocessing:

```ruby
# Pipeline: while GPU processes image N, CPU prepares image N+1
current_image = preprocess(image_path)

iterations.times do |i|
  # Copy to GPU and start async inference
  copy_to_device_async(current_image, stream)
  engine.enqueue

  # Prepare next image on CPU while GPU is busy
  next_image = preprocess(image_path) if i < iterations - 1

  # Wait for inference and get results
  engine.stream_synchronize
  outputs = copy_from_device(output_ptr)

  current_image = next_image
end
```
