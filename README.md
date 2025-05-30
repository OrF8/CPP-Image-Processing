# 🖼️ CPP Image Processing
This repository contains the implementation of Exercise 4 from the course [**67315 - Programming Workshop in C & C++**](https://shnaton.huji.ac.il/index.php/NewSyl/67315/2/2024/) at The Hebrew University Of Jerusalem ([HUJI](https://en.huji.ac.il/)). \
The project focuses on building a simple image classification system using a Multi-Layer Perceptron (MLP) neural network implemented in C++. 
> 🎓 Final Grade: **105** _(out of 105)_

# 📦 Project Structure
````
CPP-Image-Processing/
├── Activation.cpp        # Implementation of activation functions
├── Activation.h          # Declarations for activation functions
├── Dense.cpp             # Implementation of Dense layer
├── Dense.h               # Declarations for Dense layer
├── MlpNetwork.cpp        # Implementation of the MLP network
├── MlpNetwork.h          # Declarations for the MLP network
├── Matrix.cpp            # Implementation of the Matrix class
├── Matrix.h              # Declarations for the Matrix class
├── main.cpp              # Entry point of the application
├── Makefile              # Build configuration
├── plot_img.py           # Python script for visualizing images
├── images/               # Directory containing input images
├── parameters/           # Directory containing model parameters
├── LICENSE               # MIT License
└── README.md             # Project documentation
````

# 🚀 Getting Started
## Prerequisites
- C++ compiler supporting C++11 or later (e.g., `g++`)
- Python 3.x (for running `plot_img.py`)​
- Python packages: `matplotlib`, `numpy`
## Building the Project
1. Clone the repository:
   ````
   git clone https://github.com/OrF8/CPP-Image-Processing.git
   cd CPP-Image-Processing
   ````
2. Compile the project:
   ````
   make
   ````
   This will generate an executable named `mlp_classifier`.
## Running the Application
````
./mlp_classifier
````
The application will process the images located in the `images/` directory using the MLP network with parameters from the `parameters/` directory.
## Visualizing Images
To visualize an image using the provided Python script:​
````
python3 plot_img.py path_to_image
````
Replace `path_to_image` with the path to the image you wish to visualize.

# 📄 License
This project is licensed under the MIT License – see the [**LICENSE**](https://github.com/OrF8/CPP-Image-Processing/blob/main/LICENSE) file for details.



















