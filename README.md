# Input Convex Lipschitz RNN: A Fast and Robust Approach for Engineering Tasks

Zihao Wang and Zhe Wu </br>
Paper: https://arxiv.org/abs/2401.07494 (We are working on the update of the manuscript) </br>

Requires: Python 3.11.3, Tensorflow Keras 2.13.0, Numpy, Sklearn, Pickle, h5py, hdf5storage </br>

### File description:
* In the CSTR Modeling folder, we present a Continuous Stirred Tank Reactor (CSTR) example to demonstrate and compare the performance of various neural network architectures on modeling systems governed by Ordinary Differential Equations (ODEs). These include conventional Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), Lipschitz Recurrent Neural Networks (LRNNs), Input Convex Recurrent Neural Networks (ICRNNs), and our proposed Input Convex Lipschitz Recurrent Neural Networks (IC-L-RNNs). The evaluation is conducted in the presence of additive Gaussian noise.
* In the Energy System folder, similarly, we present an energy system example, i.e., a waste heat recovery system, to demonstrate and compare the performance of various neural network architectures on modeling dynamic system. The evaluation is conducted in the presence of additive Gaussian noise. The "test_collected_data.mat" stores the training data. For this task, each input sample will consist of 5 different U, i.e., the control actions, and each output will consist of 5 different Y, i.e., the predicted states, in the sequence of time. In each subfolder, the "Energy_MODEL_predict.py" will take in one sample from a MAT file, i.e., "sample.mat", and output the predicted states in another MAT file, i.e., "MODEL_prediction.mat". We provided a sample file, i.e., "sample.mat", as one test case.

### Acknowledgement
Some codes on power iteration method are modified based on:
* Serrurier, M., Mamalet, F., González-Sanz, A., Boissin, T., Loubes, J. M., & Del Barrio, E. (2021). 
  Achieving robustness in classification using optimal transport with hinge regularization. 
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 505-514).

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
