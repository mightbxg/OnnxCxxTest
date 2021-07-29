#include "onnx_test.h"
#include <cuda_provider_factory.h>
#include <iostream>

#include <TestFuncs/TicToc.hpp>

using namespace std;

namespace {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

} //namespace

namespace Ort {

void modelTest(const std::string& fn_model, const cv::Mat& input, cv::Mat& output, bool use_cuda)
{ // @ref https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp

    // environment and options
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "SuperResolution");
    Ort::SessionOptions session_options;
    if (use_cuda) {
        // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h#L13
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    }
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // load model and create session
    Ort::Session session(env, fn_model.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // model info
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_dims[0] = output_dims[0] = 1;
    vector<const char*> input_names { input_name };
    vector<const char*> output_names { output_name };

    // input & output data
    CV_Assert(input.type() == CV_8UC1);
    CV_Assert(input.rows == input_dims[2] && input.cols == input_dims[3]);
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0);
    cv::Mat result(output_dims[2], output_dims[3], CV_32FC1);
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    vector<Ort::Value> input_tensors, output_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
    output_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, result.ptr<float>(), result.total(), output_dims.data(), output_dims.size()));

    // inference
    for (int i = 0; i < 10; ++i) {
        dbg::TicToc::ScopedTimer st("onnx", true);
        session.Run(Ort::RunOptions { nullptr }, input_names.data(), input_tensors.data(), 1,
            output_names.data(), output_tensors.data(), 1);
    }

    cv::convertScaleAbs(result, output, 255.0);
}

} //namespace Ort
