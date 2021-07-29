#include <iostream>
#include <memory>

#include "cvdnn_test.h"
#include "onnx_test.h"
#include "torch_test.h"

using namespace std;
using namespace cv;

int main()
{
    const bool use_cuda = true;
    const string fp_data = "../data/";
    const string fn_image = fp_data + "cat.jpg";
    const string fn_torch_model = fp_data + "super_resolution.pt";
    const string fn_onnx_model = fp_data + "super_resolution.onnx";

    auto image = imread(fn_image, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "cannot load image: " << fn_image << endl;
        return -1;
    }

    // torch
    Mat result_torch;
    {
        auto input = torch::toTensor(image);
        torch::Tensor output_torch;
        torch::modelTest(fn_torch_model, input, output_torch, use_cuda);
        result_torch = torch::toMat(output_torch);
        imshow("result_torch", result_torch);
    }

    // cvdnn
    Mat result_cvdnn;
    {
        dnn::Backend backend = dnn::DNN_BACKEND_OPENCV;
        dnn::Target target = dnn::DNN_TARGET_CPU;
        if (use_cuda) {
            backend = dnn::DNN_BACKEND_CUDA;
            target = dnn::DNN_TARGET_CUDA;
        }
        cv::modelTest(fn_onnx_model, image, result_cvdnn, backend, target);
        imshow("result_cvdnn", result_cvdnn);
    }

    // onnx
    Mat result_onnx;
    Ort::modelTest(fn_onnx_model, image, result_onnx, use_cuda);
    imshow("result_onnx", result_onnx);

    waitKey(0);
}
