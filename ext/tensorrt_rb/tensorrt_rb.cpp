#include <rice/rice.hpp>
#include <rice/stl.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>

using namespace Rice;

class TRTLogger : public nvinfer1::ILogger {
public:
    bool verbose = false;

    void log(Severity severity, const char* msg) noexcept override {
        if (verbose || severity <= Severity::kWARNING) {
            const char* level = "";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
                case Severity::kERROR: level = "ERROR"; break;
                case Severity::kWARNING: level = "WARNING"; break;
                case Severity::kINFO: level = "INFO"; break;
                case Severity::kVERBOSE: level = "VERBOSE"; break;
            }
            if (verbose || severity <= Severity::kWARNING) {
                fprintf(stderr, "[TensorRT %s] %s\n", level, msg);
            }
        }
    }
};

class TRTEngine {
private:
    TRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<void*> bindings;
    cudaStream_t stream;

public:
    TRTEngine(const std::string& engine_path, bool verbose = false) {
        logger.verbose = verbose;

        std::ifstream file(engine_path, std::ios::binary);

        if (!file) {
            throw std::runtime_error("Failed to open engine file: " + engine_path);
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        runtime.reset(nvinfer1::createInferRuntime(logger));

        if (!runtime) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        engine.reset(runtime->deserializeCudaEngine(buffer.data(), size));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine");
        }

        context.reset(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }

        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }

        int num_tensors = engine->getNbIOTensors();

        bindings.resize(num_tensors, nullptr);
    }

    ~TRTEngine() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }

    int num_io_tensors() const {
        return engine->getNbIOTensors();
    }

    std::string get_tensor_name(int index) const {
        return engine->getIOTensorName(index);
    }

    bool is_input(const std::string& name) const {
        return engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    }

    std::vector<int64_t> get_tensor_shape(const std::string& name) const {
        auto dims = engine->getTensorShape(name.c_str());
        std::vector<int64_t> shape;
        for (int i = 0; i < dims.nbDims; i++) {
            shape.push_back(dims.d[i]);
        }
        return shape;
    }

    size_t get_tensor_bytes(const std::string& name) const {
        auto dims = engine->getTensorShape(name.c_str());
        size_t bytes = sizeof(float);
        for (int i = 0; i < dims.nbDims; i++) {
            bytes *= dims.d[i];
        }
        return bytes;
    }

    void set_tensor_address(const std::string& name, uint64_t ptr) {
        void* addr = reinterpret_cast<void*>(ptr);
        context->setTensorAddress(name.c_str(), addr);

        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            if (name == engine->getIOTensorName(i)) {
                bindings[i] = addr;
                break;
            }
        }
    }

    bool execute() {
        return context->executeV2(bindings.data());
    }

    bool enqueue() {
        return context->enqueueV3(stream);
    }

    void memcpy_htod_async(uint64_t dst, const float* src, size_t count) {
        cudaMemcpyAsync(reinterpret_cast<void*>(dst), src, count * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    void memcpy_dtoh_async(float* dst, uint64_t src, size_t count) {
        cudaMemcpyAsync(dst, reinterpret_cast<void*>(src), count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }

    void stream_synchronize() {
        cudaStreamSynchronize(stream);
    }

    uint64_t get_stream() const {
        return reinterpret_cast<uint64_t>(stream);
    }
};

extern "C" void Init_tensorrt_rb() {
    Module rb_mTensorRT = define_module("TensorRT");

    define_class_under<TRTEngine>(rb_mTensorRT, "Engine")
        .define_constructor(Constructor<TRTEngine, const std::string&, bool>(),
            Arg("engine_path"), Arg("verbose") = false)
        .define_method("num_io_tensors", &TRTEngine::num_io_tensors)
        .define_method("get_tensor_name", &TRTEngine::get_tensor_name)
        .define_method("is_input?", &TRTEngine::is_input)
        .define_method("get_tensor_shape", &TRTEngine::get_tensor_shape)
        .define_method("get_tensor_bytes", &TRTEngine::get_tensor_bytes)
        .define_method("set_tensor_address", &TRTEngine::set_tensor_address)
        .define_method("execute", &TRTEngine::execute)
        .define_method("enqueue", &TRTEngine::enqueue)
        .define_method("stream_synchronize", &TRTEngine::stream_synchronize)
        .define_method("get_stream", &TRTEngine::get_stream);
}
