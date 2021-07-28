cmake_minimum_required(VERSION 3.10)

project(OnnxCxxTest)

find_package(onnxruntime REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(t
    main.cpp
    torch_test.h
    torch_test.cpp
    cvdnn_test.h
    cvdnn_test.cpp)
target_link_libraries(t PRIVATE onnxruntime ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_features(t PRIVATE cxx_std_17)
