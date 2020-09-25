# A quick-use package for speech enhancement based on our DIHARD18 system
Original founder: @staplesinLA

Major contributor: @nryant @mmmaat(many thanks!)
			
The repository provides tools to reproduce the enhancement results of the
speech preprocessing part of our DIHARD18 system[1]. The deep-learning based
denoising model is trained on 400 hours of English and Mandarin audio; for full
details see [1,2,3]. Currently the tools accept 16 kHz, 16-bit monochannel
WAV files. Please convert the audio format in advance.

Additionally, this package integrates a voice activity detection (VAD) module
based on [py-webrtcvad](https://github.com/wiseman/py-webrtcvad), which provides a Python interface to the
[WebRTC](https://webrtc.org/) VAD. The default parameters are tuned on the
development set of DIHARD18.

[1] Sun, Lei, et al. "Speaker Diarization with Enhancing Speech for the
First DIHARD Challenge." Proc. Interspeech 2018 (2018):
2793-2797. [PDF](http://home.ustc.edu.cn/~sunlei17/pdf/lei_IS2018.pdf)

[2] Gao, Tian, et al. "Densely connected progressive learning for
lstm-based speech enhancement." 2018 IEEE International Conference on
Acoustics, Speech and Signal Processing
(ICASSP). IEEE, 2018. [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461861)

[3] Sun, Lei, et al. "Multiple-target deep learning for LSTM-RNN based
speech enhancement." 2017 Hands-free Speech Communications and
Microphone Arrays (HSCMA). IEEE,
2017. [PDF](http://home.ustc.edu.cn/~sunlei17/pdf/MULTIPLE-TARGET.pdf)


## Main Prerequisites

* [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python?tabs=cntkpy26)
* [webrtcvad](https://github.com/wiseman/py-webrtcvad)
* [Numpy](https://github.com/numpy/numpy)
* [Scipy](https://github.com/scipy/scipy)
* [Librosa](https://github.com/librosa/librosa)
* [Wurlitzer](https://github.com/minrk/wurlitzer)
* [joblib](https://github.com/joblib/joblib)

## How to use it?

1. Install all dependencies (Note that you need to have Python and pip
   already installed on your system) :

        sudo apt-get install openmpi-bin
        pip install numpy scipy librosa
        pip install cntk-gpu
        pip install webrtcvad
        pip install wurlitzer
        pip install joblib

   Make sure the CNTK engine installed successfully by querying its version:

        python -c "import cntk; print(cntk.__version__)"

2. Download the speech enhancement repository :

        git clone https://github.com/staplesinLA/denoising_DIHARD18.git
	
3. Install the pretrained model:

        cd denoising_DIHARD18
	./install_model.sh

4. Specify parameters in ``run_eval.sh``:

    * For the speech enhancement tool:

            WAV_DIR=<path to original wavs>
            SE_WAV_DIR=<path to output dir>
            USE_GPU=<true|false, if false use CPU, default=true>
            GPU_DEVICE_ID=<GPU device id on your machine, default=0>
            TRUNCATE_MINUTES=<audio chunk length in minutes, default=10>

      We recommend using a GPU for decoding as it's much faster than CPU.
      If decoding fails with a ``CUDA Error: out of memory`` error, reduce the
      value of ``TRUNCATE_MINUTES``.

    * For the VAD tool:

            VAD_DIR=<path to output dir>
            HOPLENGTH=<duration in milliseconds of VAD frame size, default=30>
            MODE=<WebRTC aggressiveness, default=3>
            NJOBS=<number of parallel processes, default=1>

5. Execute ``run_eval.sh``:

        ./run_eval.sh

### Use within docker

1. Install [docker](https://docs.docker.com/install/linux/docker-ee/ubuntu)

2. Install [nvidia docker](https://github.com/nvidia/nvidia-docker), a
   plugin to use your GPUs within docker

3. Build the image using the provided ``Dockerfile``:

        docker build -t dihard18 .

4. Run the evaluation script within docker with the following commands:

        docker run -it --rm --runtime=nvidia -v /abs/path/to/dihard/data:/data dihard18 /bin/bash
        # you are now in the docker machine
        ./run_eval.sh  # before launcing the script you can edit it to modify the parameters

   * The option ``--runtime=nvidia`` enables the use of GPUs within docker

   * The option ``-v /absolute/path/to/dihard/data:/data`` mounts the
     folder where the data are stored into Docker in the ``/data``
     folder. The directory ``/absolute/path/to/dihard/data`` **must
     contain** a ``wav/`` subdirectory. The results will be stored in
     the directories ``wav_pn_enhanced/`` and ``vad/``.


## Details

1. Speech enhancement model

   The scripts accept 16 kHz, 16-bit monochannel WAV files. Please convert the
   audio format in advance. To easily rebuild the waveform, the input feature
   is log-power spectrum (LPS). As the model has dual outputs including "IRM"
   and "LPS", the final used component is the "IRM" target which directly
   applies a mask to the original speech. Compared with "LPS" output, it can
   yield better speech intelligibility and fewer distortions.

2. VAD module

   The optional parameters of WebRTC VAD are aggressiveness mode (default=3)
   and hop length (default=30 ms). The default settings are tuned on the
   development set of the [First DIHARD challenge](https://coml.lscp.ens.fr/dihard/2018/index.html).
   For the development set, here is the comparison between original speech
   and processed speech in terms of VAD metrics:

   | VAD(default) | Original_Dev | Processed_Dev |
   | ------       | ------       | ------        |
   | Miss         | 11.85        | 7.21          |
   | FA           | 6.12         | 6.17          |
   | Total        | 17.97        | 13.38         |

   And the performance on the evaluation set:

   | VAD(default) | Original_Eval | Processed_Eval |
   | ------       | ------        | ------         |
   | Miss         | 17.49         | 8.89           |
   | FA           | 6.36          | 6.4            |
   | Total        | 23.85         | 15.29          |


3. Effectiveness

   The contribution of a single sub-module on the final speaker diarization
   performance is too trivial to analyze. However, it can be seen clearly that
   the enhancement based pre-processing is beneficial to at least VAD
   performance. Users can also tune the default VAD parameters to obtain a
   desired trade-off between Miss and False Alarm rates.
