A quick-use package of speech enhancement model in our DIHARD18 system:
----
The repository contains python tools of our speech denoising model in our DIHARD18 system. The model is trained with 400-hour mixing
data of both English and Chinese. The model architecture is presented in [Paper](http://home.ustc.edu.cn/~sunlei17/pdf/lei_IS2018.pdf):Speaker Diarization with Enhancing Speech for the First DIHARD Challenge,
which combine previous techniques of [PL](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461861) and [MTL](http://home.ustc.edu.cn/~sunlei17/pdf/MULTIPLE-TARGET.pdf).<br> 

Also, it integrates a vad interface using python-webrtcvad.

**Main Prerequisites:**<br>
>[CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python?tabs=cntkpy26): python version<br> 
[webrtcvad](https://github.com/wiseman/py-webrtcvad)<br> 
[Numpy](https://github.com/numpy/numpy)<br> 
[Scipy](https://github.com/scipy/scipy)<br> 
[Librosa](https://github.com/librosa/librosa)<br> 
 
**How to use it ?**<br> 
1. Install prerequisites.
2. Specify necessary parameters in **run_eval.sh**, such as data dictionaries, GPU id and so on.
3. Direct run it.
