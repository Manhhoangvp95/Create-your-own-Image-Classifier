# AI Programming with Python Project
## Deep Learning
### Create your own Image Classifier

#### Overview
This project contains code for an image classifier developed with PyTorch, which can be converted into a command-line application.

#### Installation
This project requires Python 3 and the following Python libraries to be installed:
- PyTorch
- ArgParse
- PIL
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You will also need to have software installed to run and execute an iPython Notebook. We recommend installing Anaconda, a pre-packaged Python distribution that includes all the necessary libraries and software for this project.

#### Running the Project
To run this project, follow these steps:

1. Clone this repository to your local machine.
2. In a terminal or command window, navigate to the project's top-level directory (the one containing this README).
3. To open the Jupyter Notebook, run one of the following commands:
   - `ipython notebook "Image Classifier Project.ipynb"`
   - `jupyter notebook "Image Classifier Project.ipynb"`
   
   This will open the iPython Notebook in your browser.
   
4. Alternatively, you can run the project from the command line using the following commands:
   - To train a new network on a dataset:
     ```
     python train.py --data_dir="flowers/" --arch="densenet121" --dropout=0.2 --hidden_units=4096 --learning_rate=0.001 --epochs=4 --gpu
     ```
     - You can specify options such as saving checkpoints, choosing the architecture, setting hyperparameters, and using GPU for training.
   - To predict the flower name from an image:
     ```
     python predict.py --image="flowers/test/10/image_07090.jpg" --topk=5 --checkpoint="checkpoint.pth" --labels="cat_to_name.json" --gpu
     ```
     - You can also specify options like returning the top K most likely classes, using a mapping of categories to real names, and using GPU for inference.

Make sure to replace the placeholders in the commands with the actual data directory, image path, or checkpoint file you want to use.

For more details on how to use specific options, please refer to the individual Python scripts.

Good luck with your image classifier project!