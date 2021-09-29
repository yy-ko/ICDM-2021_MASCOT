# MASCOT
## Introduction
In recent years, quantization methods have successfully accelerated the training of large DNN models by reducing the level of precision in computing operations (e.g., forward/backward passes) without sacrificing its accuracy. In this work, therefore, we attempt to apply such a quantization idea to the popular Matrix factorization (MF) methods to deal with  the growing scale of models and datasets in recommender systems. However, to our dismay, we observe that the state-of-the-art quantization methods are not effective in the training of MF models, unlike their successes in the training of DNN models. To this phenomenon, we posit that two 
distinctive features in training MF models could explain the difference: (i) training MF models is much more memory-intensive than training DNN models, and (ii) the quantization errors across users and items in recommendation are not uniform. From these observations, then, we develop a novel quantization framework for MF models, named as MASCOT, employing novel strategies to successfully address two aforementioned unique features in the training of MF models. The comprehensive evaluation using four real-world datasets demonstrates that MASCOT improves the training performance of MF models by about 45%, compared to the training without quantization, while maintaining low model errors, and the strategies and implementation optimizations of MASCOT are quite effective in the training of MF models.

## Building
This project is written in standard C++ and CUDA 10.2. it can be built by running Makefile in the source code directory.

## Run
Run executable file by:  
  ```
  ./quantized_mf -i [train file] -y [test file] -o [output file] [options]
  ```  


Where options are as follows:    
  > -l  : The number of epochs executed during training  
  -k  : The dimensionality of latent space (64 or 128)  
  -b  : Regularization parameter for users and items  
  -a  : Initial learning rate  
  -d  : Decay factor  
  -wg : The number of warps launched during update  
  -b  : The number of threads per block  
  -ug : The number of user groups  
  -ig : The number of item groups  
  -r  : Sampling ratio  
  -it : Error estimate period  
  -e  : Error threshold  
  -rc : Whether to save reconstructed testset matrix  
  -v  : MF version to run  
  
It is recommended to tune the number of threads using -wg options to maximize the performance.  
We used an RTX 2070 GPU for our experiments and set the number of warps to 2,048 (k = 128), 2,304 (k = 64)  
Other parameter settings are described in the paper.  

## Datasets

In our experiments, we used four real-world datasets for training and testing.  
In the case of [ML10M](https://grouplens.org/datasets/movielens/10m/) and [ML25M](https://grouplens.org/datasets/movielens/25m/), we divide the training and test set 8:2 for 5-cross validation.  
For [Netflix](https://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a) and [Yahoo!Music](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&did=48), we just use the provided training and test sets.  



<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/datasets.png" width="490" height="180">  


## Test

We provide pre-trained models ([.zip](https://figshare.com/s/9a54e7389d491688e0cc)) and you can test it as follows:  

  ```
  ./test_mf -i [pre-trained model file] -y [test file] -v [mf version]
  ```  

## Experimental results  
First, We compare MASCOT and three state-of-the-art quantization methods in terms of training time and the model error. **(RQ1~2)**  
Existing quantization methods are as follows :
  - [[ICLR '18](https://arxiv.org/abs/1710.03740)] Mixed Precision Training (MPT)
  - [[ICML '20](http://proceedings.mlr.press/v119/rajagopal20a.html)] Muti-Precision Policy Enforced Training (MuPPET)
  - [[CVPR '20](https://ieeexplore.ieee.org/abstract/document/9157439)] Adaptive Fixed Point (AFP)  


In the next experiment, we verify the effectiveness of our strategies (m-quantization, g-switching) and optimization technique through an ablation study. **(RQ3)**  
Finally, we evaluate the hyperparameter sensitivity of MASCOT and provide the best values for each hyperparameter, maximizing the performance improvement while maintaining the model errors low. **(RQ4)**  


**RQ1. Does MASCOT improve the training performance of MF models more than existing quantization methods?**  


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/Performance%20comparison.png" width="470" height="400">




**RQ2. Does MASCOT provide the errors of MF models lower than existing quantization methods?**  


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/RMSE%20comparison.png" width="470" height="400">  


**RQ3. How effective are the strategies and optimizations of MASCOT in improving the MF model training?**  

  - Strategies of MASCOT (m-quantization, g-switching) 


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/strategies%20of%20mascot.png" width="450" height="350">


  - Optimization technique


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/quantization%20optimization.png" width="850" height="350">  



**RQ4. How sensitive are the training performance and model error of MASCOT to its hyperparameters?**  

  - Sampling ratio, error estimate period 


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/hyperparameter%20sensitivity.png" width="850" height="380">  


  - The number of groups

<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/hyperparameter%20sensitivity2.png" width="850" height="350">  

You can produce those result using following commands :  

RQ 1~2 :  
  - MASCOT  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e 20 -s 0.05 -it 2 -v 1
    ```  
  - MPT  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 4
    ```  
  - MuPPET  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -s 0.05 -v 3
    ```  
  - AFP  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 2
    ```  
  - FP32  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 5
    ```  


RQ 3:  
  - Strategies of MASCOT  
    - FP32  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 5
    ```  
    - MASCOT-N1  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 7
    ```  
    - MASCOT-N2  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e [values] -s 0.05 -it 4 -v 8
    ```- The optimal error threshold (ML10M, ML25M, Netflix, Yahoo!Music) :  10, 12.5, 1.35, 20
    - MASCOT  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e 20 -s 0.05 -it 2 -v 1
    ```  


  - Optimization technique  
    - MASCOT-naive  
      ```
      ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e 20 -s 0.05 -it 2 -v 6
      ```
    - MASCOT-opt  
      ```
      ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e 20 -s 0.05 -it 2 -v 1
      ```

RQ 4:  
  - Sampling ratio, error estimate period, # of groups  
	  - MASCOT  
    ```
    ./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e [values] -s [values] -it 2 -v 1
    ```  

