#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <TestFuncs/TicToc.hpp>

using namespace std;

namespace {

torch::Tensor toTensor(const cv::Mat& image)
{
    CV_Assert(image.type() == CV_8UC1);
    return torch::from_blob(image.data,
        { 1, 1, image.rows, image.cols }, torch::kByte)
        .toType(torch::kFloat32)
        .mul(1.f / 255.f);
}

cv::Mat toMat(const torch::Tensor& tensor)
{
    using namespace torch;
    Tensor t = tensor.mul(255.f).clip(0, 255).toType(kU8).to(kCPU).squeeze();
    CV_Assert(t.sizes().size() == 2);
    return cv::Mat(t.size(0), t.size(1), CV_8UC1, t.data_ptr()).clone();
}

void torchTest(const string& fn_model, const torch::Tensor& input, torch::Tensor& output, bool use_cuda)
{
    using namespace torch;
    const auto device = use_cuda ? kCUDA : kCPU;

    jit::script::Module module;
    try {
        module = jit::load(fn_model);
        module.to(device);
    } catch (const c10::Error& e) {
        cerr << "failed to load the model from: " << fn_model << "\n"
             << e.what() << "\n";
        return;
    }

    Tensor _input = input.to(device);
    for (int i = 0; i < 10; ++i) {
        dbg::TicToc::ScopedTimer st("torch", true);
        output = module.forward({ _input }).toTensor().to(kCPU);
    }
    output = output.to(kCPU);
}

} //namespace

int main()
{
    using namespace cv;
    const bool use_cuda = false;
    const string fn_image = "../cat.jpg";
    const string fn_torch_model = "../super_resolution.pt";

    auto image = imread(fn_image, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "cannot load image: " << fn_image << endl;
        return -1;
    }
    auto input = toTensor(image);

    // torch
    torch::Tensor output_torch;
    torchTest(fn_torch_model, input, output_torch, use_cuda);
    Mat img_torch = toMat(output_torch);
    imshow("img_torch", img_torch);

    waitKey(0);
}