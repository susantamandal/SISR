# SISR
Project is Single Image Super Resolution using Multi Image Feature Fusion.
Multi-Image Feature Fusion Generative adversarial network(MFFGAN) is implemented.

Step 1: Create the working directory on google drive.

Step 2: Upload all the folders,.ipynb and .py files on the working directory.

Step 3: Create training input folder and validation input folder on google drive which contain the training and validation sets respectively.

Step 4: Open command_page.ipynb file on google colaboratory and run the commands one by one.

Note: 1) main.py takes user input parameters from  command_page.ipynb  file and calls mffgan model, 
      2) mffgan.py runs training or validation algorithm based on user inputs.
      3) In model_list.py  G1 and D1 are used for first phase(Gray scaled Images) and G2 and D2 are used for second phase(RGB images).
      4) Utility functions are used to run the mffgan models.
      5) Training output images and validation output images are saved in Checkpoints, Training_outputs and Validation_outputs respectively.
