A quick-use package of speech enhancement model in our DIHARD18 system:
----
The repository provides tools to reproduce the enhancement results of the speech preprocessing part in our DIHARD18 system[1]. The deep-learning based denoising model is trained with 400-hour mixing data of both English and Mandarin. The model details can be found in [1,2,3]. Currently the tools accept 16K, 16-bit mono audios, please convert the audio format in advance.

Additionally, this package also integrates a VAD module based on ‘py-webrtcvad’ which provides a python interface to the WebRTC Voice Activity Detector (VAD). The default parameters are tuned on the development set of DIHARD18.

[1]Sun, Lei, et al. "Speaker Diarization with Enhancing Speech for the First DIHARD Challenge." Proc. Interspeech 2018 (2018): 2793-2797.[PDF](http://home.ustc.edu.cn/~sunlei17/pdf/lei_IS2018.pdf)

[2]Gao, Tian, et al. "Densely connected progressive learning for lstm-based speech enhancement." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018. [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461861)

[3]Sun, Lei, et al. "Multiple-target deep learning for LSTM-RNN based speech enhancement." 2017 Hands-free Speech Communications and Microphone Arrays (HSCMA). IEEE, 2017.[PDF](http://home.ustc.edu.cn/~sunlei17/pdf/MULTIPLE-TARGET.pdf)


**Main Prerequisites:**<br>
>[CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python?tabs=cntkpy26): python version<br> 
[webrtcvad](https://github.com/wiseman/py-webrtcvad)<br> 
[Numpy](https://github.com/numpy/numpy)<br> 
[Scipy](https://github.com/scipy/scipy)<br> 
[Librosa](https://github.com/librosa/librosa)<br> 
 
**How to use it ?**<br> 
1. Install prerequisites.
2. Make sure the file completeness of big model file ("model/speech_enhancement.model"). It's recommended to use "git lfs clone", or directly download it via web.
3. Specify necessary parameters in **run_eval.sh**, such as data dictionaries, GPU id and so on.
4. Direct run it: "sh run_eval.sh".

**Details**<br> 
1. Speech enhancement model<br> 
The scripts accept 16K, 16-bit mono audios. Please convert the audio format in advance. To easily rebuild the waveform, the input feature is log-power spectrum(LPS). As the model has dual outputs including "IRM" and "LPS", the final used component is the "IRM" target which directly applys a mask on the original speech. Compared with "LPS" output, it can yield better speech intelligibility and fewer distortions.

2. Vad module<br> 
The optional parameters of webrtcvad are aggressiveness mode (default=3) and hop length (default=30). The default settings are tuned on the development set of the first DIHARD challenge. 
For the development set, here is the comparison between original speech and processed speech in terms of VAD metrics:

| VAD(default) | Original_Dev| Processed_Dev |
| ------ | ------ | ------ |
| Miss | 11.85 | 7.21 |
| FA | 6.12 | 6.17 |
| Total | 17.97| 13.38|

And the performance on evaluation set goes to:<br> 

| VAD(default) | Original_Eval | Processed_Eval |
| ------ | ------ | ------ |
| Miss | 17.49 | 8.89 |
| FA | 6.36 | 6.4|
| Total | 23.85| 15.29|


3. Effectiveness<br>
The effectiveness of a sub-module to the final speaker diarization performance is too trivial to analysis. However, it can be seen clearly that the enhancement based pre-processing is beneficial to at least VAD performance. Users can also tune the default VAD parameters to obtain a desired trade-off between Miss and False Alarm.

