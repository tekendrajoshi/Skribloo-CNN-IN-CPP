#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <opencv2/opencv.hpp>   // Add this for OpenCV image loading
#include "helpers.hpp"

class SimpleCNN {
public:
    SimpleCNN();

    std::pair<Eigen::VectorXd, double> forward_pass(const Eigen::Tensor<double,3>& x, int y);
    void back_pass(double lr = 0.01);

    // Convert image file to normalized tensor (1,28,28)
    Eigen::Tensor<double, 3> convert_images(const std::string& image_path);
    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    // Weights and biases
    Eigen::Tensor<double, 4> kernel_1;
    Eigen::VectorXd bias_1;

    Eigen::Tensor<double, 4> kernel_2;
    Eigen::VectorXd bias_2;

    Eigen::MatrixXd fc1_weights;
    Eigen::VectorXd fc1_bias;

    Eigen::MatrixXd fc2_weights;
    Eigen::VectorXd fc2_bias;

    struct Cache {
        Eigen::Tensor<double,3> x;
        Eigen::Tensor<double,3> x_pad;
        Eigen::Tensor<double,3> c1;
        Eigen::Tensor<double,3> r1;
        Eigen::Tensor<double,3> p1;
        Eigen::Tensor<double,3> p1_pad;
        Eigen::Tensor<double,3> c2;
        Eigen::Tensor<double,3> r2;
        Eigen::Tensor<double,3> p2;
        Eigen::VectorXd flattened;
        Eigen::MatrixXd d1;
        Eigen::MatrixXd r3;
        Eigen::VectorXd out;
        Eigen::VectorXd prob;
        int y;
    } cache;

    // Helpers to handle relu for tensor slices
    Eigen::Tensor<double, 3> tensor_relu(const Eigen::Tensor<double, 3>& x);
    Eigen::Tensor<double, 3> tensor_relu_backward(const Eigen::Tensor<double, 3>& d_out, const Eigen::Tensor<double, 3>& x);

    // Remove padding helper
    Eigen::Tensor<double, 3> pad_back(const Eigen::Tensor<double, 3>& input);
};
