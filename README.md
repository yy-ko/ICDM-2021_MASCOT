# MASCOT: A Quantization Framework for Efficient Matrix Factorization in Recommender Systems

This repository provides a reference implementation of *MASCOT* as described in the following paper:
> MASCOT: A Quantization Framework for Efficient Matrix Factorization in Recommender Systems<br>
> Yunyong Ko, Jae-Seo Yu, Hong-Kyun Bae, Yongjun Park, Dongwon Lee, and Sang-Wook Kim<br>
> IEEE International Conference on Data Mining (IEEE ICDM 2021)<br>

This project is written in standard C++ and CUDA 10.2. It can be built by running Makefile in the source code directory.

## Usage
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


## Test Pre-trained Models

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
    ```
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
    

## Citation
Please cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{ko2021mascot,
  title={MASCOT: A Quantization Framework for Efficient Matrix Factorization in Recommender Systems},
  author={Ko, Yunyong and Yu, Jae-Seo and Bae, Hong-Kyun and Park, Yongjun and Lee, Dongwon and Kim, Sang-Wook},
  booktitle={Proceedings of the 2021 IEEE International Conference on Data Mining (ICDM)},
  pages={290--299},
  year={2021},
  organization={IEEE}
}
```

