# SISR
Project is Single Image Super Resolution using Multi Image Feature Fusion.
Multi-Image Feature Fusion Generative adversarial network(MFFGAN) was implemented.

step 1: Create Training input folder and validation input folder which contains input images.
step 2: Open command_page.ipynb file and run the commands one by one.

Note: 1) main.py takes user input parameters from  command_page.ipynb  file and calls mffgan model, 
      2) mffgan.py runs training or validation algorithm based on user inputs.
      3) In model_list.py  G1 and D1 are used for first phase(Gray scaled Images) and G2 and D2 are used for second phase(RGB images).
      4) Utility functions were used to run the mffgan models.
      5) Training output images and validation output images are saved in Checkpoints, Training_outputs and Validation_outputs respectively.
 
