#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

// Padding
Eigen::Tensor<double, 3> pad(
    const Eigen::Tensor<double, 3>& input,
    int pad = 1);

// Convolution
Eigen::Tensor<double, 3> convolve(
    const Eigen::Tensor<double, 3>& X,
    const Eigen::Tensor<double, 4>& filters,
    const Eigen::VectorXd& bias);

Eigen::Tensor<double, 3> convolve_backward(
    const Eigen::Tensor<double, 3>& d_out,
    const Eigen::Tensor<double, 3>& X,
    const Eigen::Tensor<double, 4>& filters,
    Eigen::Tensor<double, 4>& d_filters,
    Eigen::VectorXd& d_bias);

// Max Pooling
Eigen::Tensor<double, 3> maxpool(
    const Eigen::Tensor<double, 3>& input,
    int size = 2,
    int stride = 2);

Eigen::Tensor<double, 3> maxpool_backward(
    const Eigen::Tensor<double, 3>& d_out,
    const Eigen::Tensor<double, 3>& input,
    int size = 2,
    int stride = 2);

// Flatten
Eigen::VectorXd flatten(const Eigen::Tensor<double, 3>& x);
Eigen::Tensor<double, 3> flatten_backward(
    const Eigen::VectorXd& d_out,
    int C, int H, int W);

// Activations
Eigen::MatrixXd relu(const Eigen::MatrixXd& x);
Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x);

// Softmax
Eigen::VectorXd softmax(const Eigen::VectorXd& x);
Eigen::VectorXd softmax_backward(
    const Eigen::VectorXd& softmax_output,
    const Eigen::VectorXd& d_out);
Eigen::VectorXd softmax_cross_entropy_backward(
    const Eigen::VectorXd& predicted,
    int actual);

// Loss
double cross_entropy_loss(
    const Eigen::VectorXd& predicted,
    int actual);
Eigen::VectorXd cross_entropy_loss_derivative(
    const Eigen::VectorXd& predicted,
    int actual);
