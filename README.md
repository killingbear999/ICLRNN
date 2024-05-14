# Input Convex Lipschitz RNN: A Fast and Robust Approach for Engineering Tasks

Zihao Wang, P S Pravin, Zhe Wu </br>
Paper: https://arxiv.org/abs/2401.07494 </br>

**Requires: Python 3.11.3, Tensorflow Keras 2.13.0, Pyipopt, Numpy, Sklearn** </br>
File description:
* docker.pptx includes the instruction on how to install Pyipopt into Docker on your laptop. </br>
* Codes for Solar PV system experiments are available in SolarPV folder. Due to confidentiality concerns, we are sorry that the data is considered proprietary to the company and, as such, is not uploaded for public usage.
* Codes for MPC experiments are available in MPC folder. There are two subfolders. Codes in CSTR subfolder are used to model the system dynamics. Codes in MPC subfolder are used to study the performance of neural networks in NN-based MPC optimization. iclrnn_original_256_0.h5, lrnn_256_0.h5, lstm_256_0.h5, rnn_256_0.h5 are trained models to be embedded into their respective MPC files.

FYI:
* Pyipopt can be installed and run on Docker. Codes in MPC subfolder use Pyipopt.

## Citation </br>
If you find our work relevant to your research, please cite:
```
@article{wang2024input,
  title={Input Convex Lipschitz RNN: A Fast and Robust Approach for Engineering Tasks},
  author={Wang, Zihao and Pravin, PS and Wu, Zhe},
  journal={arXiv preprint arXiv:2401.07494},
  year={2024}
}
```
