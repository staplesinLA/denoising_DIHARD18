# A quick-use package of speech enhancement model in our DIHARD18 system:

The repository contains python tools for our speech denoising model in
our DIHARD18 system. The model is trained with 400-hour mixing data of
both English and Chinese. The model architecture is presented in
[Paper](http://home.ustc.edu.cn/~sunlei17/pdf/lei_IS2018.pdf): Speaker
Diarization with Enhancing Speech for the First DIHARD Challenge,
which combine previous techniques of
[PL](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461861)
and [MTL](http://home.ustc.edu.cn/~sunlei17/pdf/MULTIPLE-TARGET.pdf).

Also, it integrates a vad interface using python-webrtcvad.


## Main Prerequisites

* [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python?tabs=cntkpy26):
  python version
* [webrtcvad](https://github.com/wiseman/py-webrtcvad)
* [Numpy](https://github.com/numpy/numpy)
* [Scipy](https://github.com/scipy/scipy)
* [Librosa](https://github.com/librosa/librosa)


## How to use it ?

### Standard installation

1. Install prerequisites.
2. Make sure the file completeness of big model file
   ("model/speech_enhancement.model"). It's recommended to use "git
   lfs clone", or directly download it via web.
3. Specify necessary parameters in **run_eval.sh**, such as data
   dictionaries, GPU id and so on.
4. Direct run it: "sh run_eval.sh".


### Use within docker

1. Install [docker](https://docs.docker.com/install/linux/docker-ee/ubuntu)
2. Install [nvidia docker](https://github.com/nvidia/nvidia-docker), a
   plugin to use your GPUs within docker
3. Pull the docker image: ``docker pull XXX/XXX:latest``
4. Run the evaluation script with the following command:

        docker run --rm --runtime=nvidia -v /absolute/path/to/dihard/data:/data XXX/XXX \
            /bin/bash -c "source activate dihard18 && ./run_eval.sh"

   * The option ``--runtime=nvidia`` enables the use of GPUs within docker
   * The option ``-v /absolute/path/to/dihard/data:/data`` mounts the
     folder where are stored the data into docker in the ``/data``
     folder. The directory ``/absolute/path/to/dihard/data`` must
     contains a ``wav`` subdirectory. The results will be stored on
     the subfolder ``wav_pn_enhanced``.

5. (optional) Instead of pull the image from docker hub, you can build
   the image using the provided ``Dockerfile``:

        docker build -t dihard18 .

    Once the image is built, launch the command in step 4, replacing
    ``XXX/XXX`` by ``dihard18``.


## Details

1. Speech enhancement model

   The scripts accept 16K, 16-bit mono audios. Please convert the
   audio format in advance. To easily rebuild the waveform, the input
   feature is log-power spectrum (LPS). As the model has dual outputs
   including "IRM" and "LPS", the final used component is the "IRM"
   target which directly applys a mask on the original
   speech. Compared with "LPS" output, it can yield better speech
   intelligibility and fewer distortions.

2. Vad module

   The optional parameters of webrtcvad are aggressiveness mode
   (default=3) and hop length (default=30). The default settings are
   tuned on the development set of the first DIHARD challenge.  For
   the development set, here is the comparison between original speech
   and processed speech in terms of VAD metrics:

   | VAD(default) | Original_Dev | Processed_Dev |
   | ------       | ------       | ------        |
   | Miss         | 11.85        | 7.21          |
   | FA           | 6.12         | 6.17          |
   | Total        | 17.97        | 13.38         |

   And the performance on evaluation set goes to:

   | VAD(default) | Original_Eval | Processed_Eval |
   | ------       | ------        | ------         |
   | Miss         | 17.49         | 8.89           |
   | FA           | 6.36          | 6.4            |
   | Total        | 23.85         | 15.29          |


3. Effectiveness

   The effectiveness of a sub-module to the final speaker diarization
   performance is too trivial to analysis. However, it can be seen
   clearly that the enhancement based pre-processing is beneficial to
   at least VAD performance. Users can also tune the default VAD
   parameters to obtain a desired trade-off between Miss and False
   Alarm.
