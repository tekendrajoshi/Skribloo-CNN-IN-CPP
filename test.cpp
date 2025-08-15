#include "SimpleCNN.hpp"
#include <iostream>

// Assuming test_single_image is declared somewhere accessible
void test_single_image(SimpleCNN& model, const std::string& image_path, int actual_label) {
    Eigen::Tensor<double, 3> input_tensor = model.convert_images(image_path);
    auto [probabilities, loss] = model.forward_pass(input_tensor, actual_label);

    int predicted_class = 0;
    probabilities.maxCoeff(&predicted_class);
    double confidence = probabilities(predicted_class);

    std::cout << "Actual class: " << actual_label 
              << ", Predicted class: " << predicted_class 
              << " with confidence: " << confidence << std::endl;
}
int main() {
    SimpleCNN model;

    // Load saved model parameters
    model.load("simple_cnn_model.txt");

    // Test on one or more images
    test_single_image(model, "/home/vishal/Vikash/Projects/Final_project/Quick_Draw/quickdraw_png/apple/apple_0991.png", 1);
    test_single_image(model, "/home/vishal/Vikash/Projects/Final_project/Quick_Draw/quickdraw_png/bicycle/bicycle_0009.png", 2);
    test_single_image(model, "/home/vishal/Vikash/Projects/Final_project/Quick_Draw/quickdraw_png/book/book_0014.png", 3); 
    test_single_image(model, "/home/vishal/Vikash/Projects/Final_project/Quick_Draw/quickdraw_png/car/car_0006.png", 4);
    test_single_image(model, "/home/vishal/Vikash/Projects/Final_project/Quick_Draw/quickdraw_png/clock/clock_0011.png", 7);

    return 0;
}
