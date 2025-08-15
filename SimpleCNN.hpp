#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <opencv2/opencv.hpp>   // For image loading
#include "helpers.hpp"

// ----------------------
// SimpleCNN: A small Convolutional Neural Network
// ----------------------
class SimpleCNN {
public:
    SimpleCNN();  // Constructor: initialize weights and biases

    // Forward pass through the network
    // Input: x (C x H x W tensor), y (true label)
    // Returns: predicted probabilities vector and cross-entropy loss
    std::pair<Eigen::VectorXd, double> forward_pass(const Eigen::Tensor<double,3>& x, int y);

    // Backward pass (gradient descent step)
    // Updates network parameters using stored cache from forward_pass
    void back_pass(double lr = 0.01);

    // Load an image from file and convert it to normalized tensor (C x H x W)
    Eigen::Tensor<double, 3> convert_images(const std::string& image_path);

    // Save network weights to file
    void save(const std::string& filename) const;

    // Load network weights from file
    void load(const std::string& filename);

private:
    // ----------------------
    // Weights and biases
    // ----------------------
    Eigen::Tensor<double, 4> kernel_1; // Conv layer 1 filters
    Eigen::VectorXd bias_1;            // Conv layer 1 biases

    Eigen::Tensor<double, 4> kernel_2; // Conv layer 2 filters
    Eigen::VectorXd bias_2;            // Conv layer 2 biases

    Eigen::MatrixXd fc1_weights;       // Fully connected layer 1 weights
    Eigen::VectorXd fc1_bias;          // Fully connected layer 1 biases

    Eigen::MatrixXd fc2_weights;       // Fully connected layer 2 weights
    Eigen::VectorXd fc2_bias;          // Fully connected layer 2 biases

    // ----------------------
    // Cache: store intermediate results for backpropagation
    // ----------------------
    struct Cache {
        Eigen::Tensor<double,3> x;        // Input tensor
        Eigen::Tensor<double,3> x_pad;    // Padded input
        Eigen::Tensor<double,3> c1;       // Conv1 output
        Eigen::Tensor<double,3> r1;       // ReLU after conv1
        Eigen::Tensor<double,3> p1;       // MaxPool1 output
        Eigen::Tensor<double,3> p1_pad;   // Padded pooled output (for conv2)
        Eigen::Tensor<double,3> c2;       // Conv2 output
        Eigen::Tensor<double,3> r2;       // ReLU after conv2
        Eigen::Tensor<double,3> p2;       // MaxPool2 output
        Eigen::VectorXd flattened;        // Flattened tensor before FC layers
        Eigen::MatrixXd d1;               // FC1 pre-activation
        Eigen::MatrixXd r3;               // ReLU after FC1
        Eigen::VectorXd out;              // FC2 output (logits)
        Eigen::VectorXd prob;             // Softmax probabilities
        int y;                             // True label
    } cache;

    // ----------------------
    // Helpers for ReLU on tensors
    // ----------------------
    Eigen::Tensor<double, 3> tensor_relu(const Eigen::Tensor<double, 3>& x);  // ReLU activation
    Eigen::Tensor<double, 3> tensor_relu_backward(const Eigen::Tensor<double, 3>& d_out, const Eigen::Tensor<double, 3>& x); // ReLU gradient

    // Remove padding from tensor (used in backward pass)
    Eigen::Tensor<double, 3> pad_back(const Eigen::Tensor<double, 3>& input);
};
