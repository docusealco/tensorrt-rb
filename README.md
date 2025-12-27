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

# Query engine
engine.num_io_tensors          # Number of input/output tensors
engine.get_tensor_name(0)      # Get tensor name by index
engine.is_input?("input")      # Check if tensor is input
engine.get_tensor_shape("input") # Get tensor shape [1, 3, 640, 640]
engine.get_tensor_bytes("input") # Get tensor size in bytes

# Set memory addresses and execute
engine.set_tensor_address("input", device_ptr)
engine.execute
```
