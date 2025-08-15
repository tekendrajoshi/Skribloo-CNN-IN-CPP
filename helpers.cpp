#include "helpers.hpp"
#include <cmath>

// Padding
Eigen::Tensor<double, 3> pad(const Eigen::Tensor<double, 3>& input, int pad) {
    int C = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);

    Eigen::Tensor<double, 3> output(C, H + 2 * pad, W + 2 * pad);
    output.setZero();
    output.slice(Eigen::array<Eigen::Index, 3>{0, pad, pad},
                 Eigen::array<Eigen::Index, 3>{C, H, W}) = input;
    return output;
}

// Convolution forward
Eigen::Tensor<double, 3> convolve(
    const Eigen::Tensor<double, 3>& X,
    const Eigen::Tensor<double, 4>& filters,
    const Eigen::VectorXd& bias)
{
    int C_in = X.dimension(0);
    int H_in = X.dimension(1);
    int W_in = X.dimension(2);

    int N_filters = filters.dimension(0);
    int K = filters.dimension(2);

    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    Eigen::Tensor<double, 3> out(N_filters, H_out, W_out);
    out.setZero();

    for (int f = 0; f < N_filters; f++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                double sum = 0.0;
                for (int c = 0; c < C_in; c++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            sum += X(c, i + ki, j + kj) * filters(f, c, ki, kj);
                        }
                    }
                }
                out(f, i, j) = sum + bias(f);
            }
        }
    }
    return out;
}

// Convolution backward
Eigen::Tensor<double, 3> convolve_backward(
    const Eigen::Tensor<double, 3>& d_out,
    const Eigen::Tensor<double, 3>& X,
    const Eigen::Tensor<double, 4>& filters,
    Eigen::Tensor<double, 4>& d_filters,
    Eigen::VectorXd& d_bias)
{
    int N_filters = filters.dimension(0);
    int C_in = filters.dimension(1);
    int K = filters.dimension(2);
    int H_in = X.dimension(1);
    int W_in = X.dimension(2);

    int H_out = d_out.dimension(1);
    int W_out = d_out.dimension(2);

    Eigen::Tensor<double, 3> dX(C_in, H_in, W_in);
    dX.setZero();
    d_filters.setZero();
    d_bias.setZero();

    for (int f = 0; f < N_filters; f++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                d_bias(f) += d_out(f, i, j);
                for (int c = 0; c < C_in; c++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            d_filters(f, c, ki, kj) += X(c, i + ki, j + kj) * d_out(f, i, j);
                            dX(c, i + ki, j + kj) += filters(f, c, ki, kj) * d_out(f, i, j);
                        }
                    }
                }
            }
        }
    }
    return dX;
}

// Max Pooling forward
Eigen::Tensor<double, 3> maxpool(
    const Eigen::Tensor<double, 3>& input,
    int size,
    int stride)
{
    int C = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);

    int H_out = (H - size) / stride + 1;
    int W_out = (W - size) / stride + 1;

    Eigen::Tensor<double, 3> output(C, H_out, W_out);
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                double max_val = -1e9;
                for (int pi = 0; pi < size; pi++) {
                    for (int pj = 0; pj < size; pj++) {
                        double val = input(c, i * stride + pi, j * stride + pj);
                        if (val > max_val) max_val = val;
                    }
                }
                output(c, i, j) = max_val;
            }
        }
    }
    return output;
}

// Max Pooling backward
Eigen::Tensor<double, 3> maxpool_backward(
    const Eigen::Tensor<double, 3>& d_out,
    const Eigen::Tensor<double, 3>& input,
    int size,
    int stride)
{
    int C = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);
    int H_out = d_out.dimension(1);
    int W_out = d_out.dimension(2);

    Eigen::Tensor<double, 3> d_input(C, H, W);
    d_input.setZero();

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                double max_val = -1e9;
                int max_i = -1, max_j = -1;
                for (int pi = 0; pi < size; pi++) {
                    for (int pj = 0; pj < size; pj++) {
                        double val = input(c, i * stride + pi, j * stride + pj);
                        if (val > max_val) {
                            max_val = val;
                            max_i = pi;
                            max_j = pj;
                        }
                    }
                }
                d_input(c, i * stride + max_i, j * stride + max_j) += d_out(c, i, j);
            }
        }
    }
    return d_input;
}

// Flatten forward
Eigen::VectorXd flatten(const Eigen::Tensor<double, 3>& x) {
    int size = x.dimension(0) * x.dimension(1) * x.dimension(2);
    Eigen::VectorXd out(size);
    int idx = 0;
    for (int c = 0; c < x.dimension(0); c++) {
        for (int i = 0; i < x.dimension(1); i++) {
            for (int j = 0; j < x.dimension(2); j++) {
                out(idx++) = x(c, i, j);
            }
        }
    }
    return out;
}

// Flatten backward
Eigen::Tensor<double, 3> flatten_backward(
    const Eigen::VectorXd& d_out,
    int C, int H, int W)
{
    Eigen::Tensor<double, 3> d_input(C, H, W);
    int idx = 0;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                d_input(c, i, j) = d_out(idx++);
            }
        }
    }
    return d_input;
}

// ReLU
Eigen::MatrixXd relu(const Eigen::MatrixXd& x) {
    return x.cwiseMax(0.0);
}
Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x) {
    return (x.array() > 0).cast<double>();
}

// Softmax forward
Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd shifted = x.array() - x.maxCoeff();
    Eigen::VectorXd exp_x = shifted.array().exp();
    return exp_x / exp_x.sum();
}

// Softmax backward
Eigen::VectorXd softmax_backward(
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& d_out)
{
    double dot = s.dot(d_out);
    return s.array() * (d_out.array() - dot);
}

// Softmax + Cross-Entropy backward shortcut
Eigen::VectorXd softmax_cross_entropy_backward(
    const Eigen::VectorXd& predicted,
    int actual)
{
    Eigen::VectorXd grad = predicted;
    grad(actual) -= 1.0;
    return grad;
}

// Cross-entropy loss
double cross_entropy_loss(const Eigen::VectorXd& predicted, int actual) {
    return -std::log(predicted(actual) + 1e-15);
}
Eigen::VectorXd cross_entropy_loss_derivative(
    const Eigen::VectorXd& predicted,
    int actual)
{
    Eigen::VectorXd grad = predicted;
    grad(actual) -= 1.0;
    return grad;
}
