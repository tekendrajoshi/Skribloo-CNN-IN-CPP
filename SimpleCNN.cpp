#include "SimpleCNN.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <string>

void SimpleCNN::save(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for saving: " << filename << std::endl;
        return;
    }

    // Save kernel_1
    ofs << "kernel_1 " << kernel_1.dimension(0) << ' ' << kernel_1.dimension(1) << ' '
        << kernel_1.dimension(2) << ' ' << kernel_1.dimension(3) << '\n';
    for (int f = 0; f < kernel_1.dimension(0); ++f)
        for (int c = 0; c < kernel_1.dimension(1); ++c)
            for (int i = 0; i < kernel_1.dimension(2); ++i)
                for (int j = 0; j < kernel_1.dimension(3); ++j)
                    ofs << kernel_1(f, c, i, j) << ' ';
    ofs << '\n';

    // Save bias_1
    ofs << "bias_1 " << bias_1.size() << '\n';
    for (int i = 0; i < bias_1.size(); ++i) ofs << bias_1(i) << ' ';
    ofs << '\n';

    // Save kernel_2
    ofs << "kernel_2 " << kernel_2.dimension(0) << ' ' << kernel_2.dimension(1) << ' '
        << kernel_2.dimension(2) << ' ' << kernel_2.dimension(3) << '\n';
    for (int f = 0; f < kernel_2.dimension(0); ++f)
        for (int c = 0; c < kernel_2.dimension(1); ++c)
            for (int i = 0; i < kernel_2.dimension(2); ++i)
                for (int j = 0; j < kernel_2.dimension(3); ++j)
                    ofs << kernel_2(f, c, i, j) << ' ';
    ofs << '\n';

    // Save bias_2
    ofs << "bias_2 " << bias_2.size() << '\n';
    for (int i = 0; i < bias_2.size(); ++i) ofs << bias_2(i) << ' ';
    ofs << '\n';

    // Save fc1_weights (rows x cols)
    ofs << "fc1_weights " << fc1_weights.rows() << ' ' << fc1_weights.cols() << '\n';
    for (int i = 0; i < fc1_weights.rows(); ++i)
        for (int j = 0; j < fc1_weights.cols(); ++j)
            ofs << fc1_weights(i, j) << ' ';
    ofs << '\n';

    // Save fc1_bias
    ofs << "fc1_bias " << fc1_bias.size() << '\n';
    for (int i = 0; i < fc1_bias.size(); ++i) ofs << fc1_bias(i) << ' ';
    ofs << '\n';

    // Save fc2_weights (rows x cols)
    ofs << "fc2_weights " << fc2_weights.rows() << ' ' << fc2_weights.cols() << '\n';
    for (int i = 0; i < fc2_weights.rows(); ++i)
        for (int j = 0; j < fc2_weights.cols(); ++j)
            ofs << fc2_weights(i, j) << ' ';
    ofs << '\n';

    // Save fc2_bias
    ofs << "fc2_bias " << fc2_bias.size() << '\n';
    for (int i = 0; i < fc2_bias.size(); ++i) ofs << fc2_bias(i) << ' ';
    ofs << '\n';

    ofs.close();
}


void SimpleCNN::load(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file for loading: " << filename << std::endl;
        return;
    }

    std::string tag;
    while (ifs >> tag) {
        if (tag == "kernel_1") {
            int d0, d1, d2, d3;
            ifs >> d0 >> d1 >> d2 >> d3;
            kernel_1 = Eigen::Tensor<double, 4>(d0, d1, d2, d3);
            for (int f = 0; f < d0; ++f)
                for (int c = 0; c < d1; ++c)
                    for (int i = 0; i < d2; ++i)
                        for (int j = 0; j < d3; ++j)
                            ifs >> kernel_1(f, c, i, j);
        }
        else if (tag == "bias_1") {
            int size;
            ifs >> size;
            bias_1.resize(size);
            for (int i = 0; i < size; ++i)
                ifs >> bias_1(i);
        }
        else if (tag == "kernel_2") {
            int d0, d1, d2, d3;
            ifs >> d0 >> d1 >> d2 >> d3;
            kernel_2 = Eigen::Tensor<double, 4>(d0, d1, d2, d3);
            for (int f = 0; f < d0; ++f)
                for (int c = 0; c < d1; ++c)
                    for (int i = 0; i < d2; ++i)
                        for (int j = 0; j < d3; ++j)
                            ifs >> kernel_2(f, c, i, j);
        }
        else if (tag == "bias_2") {
            int size;
            ifs >> size;
            bias_2.resize(size);
            for (int i = 0; i < size; ++i)
                ifs >> bias_2(i);
        }
        else if (tag == "fc1_weights") {
            int rows, cols;
            ifs >> rows >> cols;
            fc1_weights.resize(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    ifs >> fc1_weights(i, j);
        }
        else if (tag == "fc1_bias") {
            int size;
            ifs >> size;
            fc1_bias.resize(size);
            for (int i = 0; i < size; ++i)
                ifs >> fc1_bias(i);
        }
        else if (tag == "fc2_weights") {
            int rows, cols;
            ifs >> rows >> cols;
            fc2_weights.resize(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    ifs >> fc2_weights(i, j);
        }
        else if (tag == "fc2_bias") {
            int size;
            ifs >> size;
            fc2_bias.resize(size);
            for (int i = 0; i < size; ++i)
                ifs >> fc2_bias(i);
        }
        else {
            std::cerr << "Unknown tag in model file: " << tag << std::endl;
            break;
        }
    }
    ifs.close();
}




SimpleCNN::SimpleCNN() {
    std::cout << "Initializing model..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());

    auto he_init = [&](int fan_in) {
        double stddev = std::sqrt(2.0 / fan_in);
        return std::normal_distribution<double>(0.0, stddev);
    };

    kernel_1 = Eigen::Tensor<double, 4>(8,1,3,3);
    {
        auto dist = he_init(3*3);
        for (int f=0; f<8; ++f)
            for (int c=0; c<1; ++c)
                for (int i=0; i<3; ++i)
                    for (int j=0; j<3; ++j)
                        kernel_1(f,c,i,j) = dist(gen);
    }
    bias_1 = Eigen::VectorXd::Zero(8);

    kernel_2 = Eigen::Tensor<double, 4>(8,8,3,3);
    {
        auto dist = he_init(3*3*8);
        for (int f=0; f<8; ++f)
            for (int c=0; c<8; ++c)
                for (int i=0; i<3; ++i)
                    for (int j=0; j<3; ++j)
                        kernel_2(f,c,i,j) = dist(gen);
    }
    bias_2 = Eigen::VectorXd::Zero(8);

    fc1_weights = Eigen::MatrixXd(392, 32);
    {
        double stddev = std::sqrt(2.0 / 392);
        std::normal_distribution<double> dist(0.0, stddev);
        for (int i=0; i<392; ++i)
            for (int j=0; j<32; ++j)
                fc1_weights(i,j) = dist(gen);
    }
    fc1_bias = Eigen::VectorXd::Zero(32);

    fc2_weights = Eigen::MatrixXd(32, 10);
    {
        double stddev = std::sqrt(2.0 / 32);
        std::normal_distribution<double> dist(0.0, stddev);
        for (int i=0; i<32; ++i)
            for (int j=0; j<10; ++j)
                fc2_weights(i,j) = dist(gen);
    }
    fc2_bias = Eigen::VectorXd::Zero(10);
}
std::pair<Eigen::VectorXd, double> SimpleCNN::forward_pass(const Eigen::Tensor<double,3>& x, int y) {
    cache.x = x;
    cache.x_pad = pad(x, 1);

    cache.c1 = convolve(cache.x_pad, kernel_1, bias_1);

    cache.r1 = tensor_relu(cache.c1);

    cache.p1 = maxpool(cache.r1);
    cache.p1_pad = pad(cache.p1, 1);

    cache.c2 = convolve(cache.p1_pad, kernel_2, bias_2);
    cache.r2 = tensor_relu(cache.c2);
    cache.p2 = maxpool(cache.r2);

    cache.flattened = flatten(cache.p2);  // VectorXd size 392

    // fc1_weights: 392 x 32, flattened: 392 x 1 vector
    // Multiply weights^T (32x392) * flattened (392x1) => (32x1)
    cache.d1 = fc1_weights.transpose() * cache.flattened;  // (32,)

    cache.d1 += fc1_bias;  // add bias (32,)

    cache.r3 = relu(cache.d1);

    // fc2_weights: 32 x 10
    // Multiply weights^T (10x32) * r3 (32x1) => (10x1)
    cache.out = fc2_weights.transpose() * cache.r3;

    cache.out += fc2_bias;

    cache.prob = softmax(cache.out);

    double loss = cross_entropy_loss(cache.prob, y);
    cache.y = y;

    return {cache.prob, loss};
}
void SimpleCNN::back_pass(double lr) {
    // Gradient of loss wrt output (10,)
    Eigen::VectorXd d_out = softmax_cross_entropy_backward(cache.prob, cache.y);

    // Gradients for fc2 weights and bias
    // cache.r3: (32,), d_out: (10,)
    // dW_fc2 = r3 * d_out^T -> (32 x 10)
    Eigen::MatrixXd dW_fc2 = cache.r3 * d_out.transpose();
    Eigen::VectorXd db_fc2 = d_out;

    // Gradient wrt fc1 output (32,)
    // dr3 = W_fc2 * d_out => (32 x 10) * (10,) = (32,)
    Eigen::VectorXd dr3 = fc2_weights * d_out;

    // Backprop through ReLU
    Eigen::VectorXd d_d1 = dr3.array() * relu_derivative(cache.d1).array();

    // Gradients for fc1 weights and bias
    // cache.flattened: (392,), d_d1: (32,)
    // dW_fc1 = flattened * d_d1^T -> (392 x 32)
    Eigen::MatrixXd dW_fc1 = cache.flattened * d_d1.transpose();
    Eigen::VectorXd db_fc1 = d_d1;

    // Gradient wrt flattened input vector (392,)
    Eigen::VectorXd d_flat = fc1_weights * d_d1;

    // Backprop flatten to tensor shape
    Eigen::Tensor<double, 3> d_p2 = flatten_backward(d_flat, cache.p2.dimension(0), cache.p2.dimension(1), cache.p2.dimension(2));

    // Maxpool 2 backward
    Eigen::Tensor<double, 3> d_r2 = maxpool_backward(d_p2, cache.r2);

    // ReLU 2 backward
    Eigen::Tensor<double, 3> d_c2 = tensor_relu_backward(d_r2, cache.c2);

    // Conv2 backward
    Eigen::Tensor<double, 4> dK2(kernel_2.dimensions());
    Eigen::VectorXd db2 = Eigen::VectorXd::Zero(bias_2.size());
    Eigen::Tensor<double, 3> d_p1_pad = convolve_backward(d_c2, cache.p1_pad, kernel_2, dK2, db2);

    // Remove padding
    Eigen::Tensor<double, 3> d_p1 = pad_back(d_p1_pad);

    // Maxpool 1 backward
    Eigen::Tensor<double, 3> d_r1 = maxpool_backward(d_p1, cache.r1);

    // ReLU 1 backward
    Eigen::Tensor<double, 3> d_c1 = tensor_relu_backward(d_r1, cache.c1);

    // Conv1 backward
    Eigen::Tensor<double, 4> dK1(kernel_1.dimensions());
    Eigen::VectorXd db1 = Eigen::VectorXd::Zero(bias_1.size());
    Eigen::Tensor<double, 3> d_x_pad = convolve_backward(d_c1, cache.x_pad, kernel_1, dK1, db1);

    // Remove padding
    Eigen::Tensor<double, 3> d_x = pad_back(d_x_pad);

    // Update parameters
    fc2_weights -= lr * dW_fc2;
    fc2_bias -= lr * db_fc2;

    fc1_weights -= lr * dW_fc1;
    fc1_bias -= lr * db_fc1;

    kernel_2 -= lr * dK2;
    bias_2 -= lr * db2;

    kernel_1 -= lr * dK1;
    bias_1 -= lr * db1;
}


// Private helper implementations

Eigen::Tensor<double, 3> SimpleCNN::tensor_relu(const Eigen::Tensor<double, 3>& x) {
    Eigen::Tensor<double, 3> out(x.dimensions());
    int C = x.dimension(0);
    int H = x.dimension(1);
    int W = x.dimension(2);
    for (int c = 0; c < C; ++c) {
        Eigen::MatrixXd mat(H, W);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                mat(i, j) = x(c, i, j);
        Eigen::MatrixXd relu_mat = relu(mat);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                out(c, i, j) = relu_mat(i, j);
    }
    return out;
}

Eigen::Tensor<double, 3> SimpleCNN::tensor_relu_backward(const Eigen::Tensor<double, 3>& d_out, const Eigen::Tensor<double, 3>& x) {
    Eigen::Tensor<double, 3> d_x(x.dimensions());
    int C = x.dimension(0);
    int H = x.dimension(1);
    int W = x.dimension(2);
    for (int c = 0; c < C; ++c) {
        Eigen::MatrixXd mat(H, W), dout_mat(H, W);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                mat(i, j) = x(c, i, j);
                dout_mat(i, j) = d_out(c, i, j);
            }
        Eigen::MatrixXd deriv = relu_derivative(mat);
        Eigen::MatrixXd grad = deriv.array() * dout_mat.array();
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                d_x(c, i, j) = grad(i, j);
    }
    return d_x;
}

Eigen::Tensor<double, 3> SimpleCNN::pad_back(const Eigen::Tensor<double, 3>& input) {
    int C = input.dimension(0);
    int H = input.dimension(1);
    int W = input.dimension(2);
    Eigen::Tensor<double, 3> output(C, H - 2, W - 2);
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < H - 2; ++i)
            for (int j = 0; j < W - 2; ++j)
                output(c, i, j) = input(c, i + 1, j + 1);
    return output;
}

// New function implementation must be **outside** other function bodies:
Eigen::Tensor<double, 3> SimpleCNN::convert_images(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);  // load grayscale image
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return Eigen::Tensor<double, 3>(1, 28, 28).setZero();
    }

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(28, 28));  // resize to 28x28

    Eigen::Tensor<double, 3> tensor(1, 28, 28);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            // Normalize pixel values from [0,255] to [0,1]
            tensor(0, i, j) = static_cast<double>(resized_img.at<uchar>(i, j)) / 255.0;
        }
    }
    return tensor;
}



