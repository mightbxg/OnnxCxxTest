#include <iostream>
#include <memory>

#include "torch_test.h"

using namespace std;
using namespace cv;

int main()
{
    const bool use_cuda = false;
    const string fn_image = "../cat.jpg";
    const string fn_torch_model = "../super_resolution.pt";

    auto image = imread(fn_image, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "cannot load image: " << fn_image << endl;
        return -1;
    }
    auto input = torch::toTensor(image);

    // torch
    torch::Tensor output_torch;
    torch::torchTest(fn_torch_model, input, output_torch, use_cuda);
    Mat img_torch = torch::toMat(output_torch);
    imshow("img_torch", img_torch);

    waitKey(0);
}
