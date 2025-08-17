# Skribloo – AI Sketch Recognition in C++ using CNN 🎨🤖

**Skribloo** is an interactive sketch recognition game inspired by Google’s QuickDraw. It allows users to draw sketches on a canvas and have a custom **Convolutional Neural Network (CNN)** implemented entirely from scratch in C++ predict the object in real-time.

This project was developed as part of the **OOP 2nd Semester Project** at **Pulchowk Campus** and provides hands-on experience in combining low-level programming efficiency with AI and deep learning concepts.

---

## Workflow

### Training Process

1. Load labeled dataset from CSV (preprocessed 28×28 images).  
2. Initialize CNN parameters: convolutional filters, fully connected weights, and biases.  
3. Forward pass: convolution → ReLU → pooling → flatten → fully connected → softmax.  
4. Compute cross-entropy loss.  
5. Backpropagation to update convolutional filters, weights, and biases.  
6. Repeat for multiple epochs (e.g., 30) tracking accuracy and loss.  
7. Save trained parameters for future inference.

---

## Acknowledgments

**Team:**  
- Tekendra Joshi  
- Vikash Pokharel  

**Lecturer:** Daya Sagar Baral  

Department of Electronics and Computer Engineering,
Institute of Engineering Pulchowk Campus  

**Dataset:** Google QuickDraw
