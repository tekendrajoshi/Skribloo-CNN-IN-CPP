#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

// ----------------------
// Padding
// ----------------------
// Adds zero-padding around the input tensor.
// Useful to preserve spatial dimensions after convolution.
Eigen::Tensor<double, 3> pad(
    const Eigen::Tensor<double, 3>& input,
    int pad = 1);

// ----------------------
// Convolution
// ----------------------
// Performs a 2D convolution of the input tensor with a set of filters.
// Adds bias to each output channel.
Eigen::Tensor<double, 3> convolve(
    const Eigen::Tensor<double, 3>& X,
    const Eigen::Tensor<double, 4>& filters,
    const Eigen::VectorXd& bias);

// Computes gradients w.r.t input, filters, and bias for backpropagation.
Eigen::Tensor<double, 3> convolve_backward(
    const Eigen::Tensor<double, 3>& d_out,
    const Eigen::Tensor<double, 3>& X,
    const Eigen::Tensor<double, 4>& filters,
    Eigen::Tensor<double, 4>& d_filters,
    Eigen::VectorXd& d_bias);

// ----------------------
// Max Pooling
// ----------------------
// Performs max pooling on the input tensor with given window size and stride.
Eigen::Tensor<double, 3> maxpool(
    const Eigen::Tensor<double, 3>& input,
    int size = 2,
    int stride = 2);

// Computes gradient of max pooling w.r.t input for backpropagation.
Eigen::Tensor<double, 3> maxpool_backward(
    const Eigen::Tensor<double, 3>& d_out,
    const Eigen::Tensor<double, 3>& input,
    int size = 2,
    int stride = 2);

// ----------------------
// Flatten
// ----------------------
// Converts a 3D tensor (C x H x W) into a 1D vector.
Eigen::VectorXd flatten(const Eigen::Tensor<double, 3>& x);

// Reshapes gradient from flattened vector back to 3D tensor for backpropagation.
Eigen::Tensor<double, 3> flatten_backward(
    const Eigen::VectorXd& d_out,
    int C, int H, int W);

// ----------------------
// Activations
// ----------------------
// Element-wise ReLU activation function.
Eigen::MatrixXd relu(const Eigen::MatrixXd& x);

// Derivative of ReLU for backpropagation.
Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x);

// ----------------------
// Softmax
// ----------------------
// Computes softmax vector for multi-class probabilities.
Eigen::VectorXd softmax(const Eigen::VectorXd& x);

// Backpropagation through softmax when combined with other layers.
Eigen::VectorXd softmax_backward(
    const Eigen::VectorXd& softmax_output,
    const Eigen::VectorXd& d_out);

// Backpropagation for softmax + cross-entropy loss.
Eigen::VectorXd softmax_cross_entropy_backward(
    const Eigen::VectorXd& predicted,
    int actual);

// ----------------------
// Loss
// ----------------------
// Computes cross-entropy loss given predicted probabilities and true label.
double cross_entropy_loss(
    const Eigen::VectorXd& predicted,
    int actual);

// Derivative of cross-entropy loss w.r.t predicted probabilities.
Eigen::VectorXd cross_entropy_loss_derivative(
    const Eigen::VectorXd& predicted,
    int actual);
