# Multimodal Micro-expression Recognition Algorithm Research


### **Ranked #Competitive Advantage** on  [Micro-expression Recognition (MER) Algorithm on CAS(ME)3 Datasets]

Multimodality is an important research topic in the field of affective computing (AC). Currently, multimodal macro-expression recognition has been researched and developed by expanding datasets.


Compared to other modalities, ECG signals are genuine and unmaskable physiological signals characterized by reliability and authenticity. They are also a relatively common and easily acquired physiological signal, as seen in hospital ECG monitors and consumer smart wearable devices (e.g., Apple Watch). They offer the advantages of portability, non-invasiveness, reliability, and high computational efficiency. The requirements for the acquisition environment are relatively simple, and they are not affected by factors such as lighting, temperature, or noise. Related research also indicates that ECG signals have been studied more extensively than other physiological signals in the context of emotion recognition. However, they have not yet been combined with micro-expressions, and challenges remain, including individual differences in how emotions affect psychological activity, significant interference (such as the impact of medical conditions on cardiac activity), and a limited number of datasets.

Therefore, we propose a multimodal tangent-weighted fusion (MTWF) micro-expression recognition network (MTWFNet) based on ECG-QRS waves for the multimodal micro-expression recognition problem based on CAS(ME)3.


The MTWFNet algorithm primarily addresses the following two aspects:
- 1.To address the issue of feature representation for ECG signals in micro-expressions, we have, for the first time, utilized the ECG-QRS band to construct a two-dimensional QRS time-domain feature matrix. This matrix not only captures the temporal and amplitude information of the QRS waveform but also incorporates information regarding the magnitude and rate of change of the QRS waveform.
- 2.The issue of multimodal fusion in micro-expression recognition. We are the first to fuse RGB video signals and ECG signals from micro-expressions, and have developed a micro-expression recognition algorithm based on tangent-weighted multimodal fusion.

Finally, we conducted extensive experimental validation on public datasets CAS(ME)3, demonstrating the feasibility and effectiveness of the algorithm.

The MTWFNet algorithm is a visual multimodal MER research method. This algorithm requires either the offline construction of handcrafted features in advance or the use of pre-processed features provided by the user.
 - Recommended GPU platform: NVIDIA GeForce RTX 4070 Ti GPU.
 - If you are using a different GPU and encounter insufficient memory, please select smaller model hyperparameters as appropriate.

#### 0. Setup the environment
-   Setup a python environment and install dependencies. Our python version == 3.12.2
```
pip install -r requirements.txt
```
#### 1. Prepare the data set
-   The raw dataset CAS(ME)3, which supports the findings of this study, is subject to licensing restrictions and cannot be made publicly available directly. It must be downloaded after submitting a request.
-   You can use the sample preprocessed dataset provided in this directory.
-   Alternatively, run the _preprocess_ script file using the following command to manually construct your own feature set.
```
python ./preprocess.py
```

#### 2. Complete the config file
In the config file, the main settings are shown in the following Table.
|Item|Description| 
|-|--|
|_gamma_|Hyperparameters indicating fusion strength|
|_sde_cde4C_|Selection of classification tasks and normalization strategies|
|_ispretrained_|Do you need to load the pre-trained model?|
|_isallparameters_|You can freeze certain layers of the structure to fine-tune the model.|
|_Epoch_|Number of training epochs|
|_DeepF_|Deep Features|
|_changelr_|Learning rate adjustment strategies can be configured according to model requirements.|
|_clasnum_|Classification Task|
|_is_augment_|Does the training set require data augmentation?|
|_train_deepfea_|Depth Emotion Feature Dimension|

#### 3. Train and Validate the model
If the above processes are prepared, then it's time for training.
```
python ./train4C.py
```
If you need to record training logs and run them in the background, run
```
nohup python -u train4C.py > ./result/your_log_file_name.log 2>&1 &
```

#### 4. Test the model
Directly run the _last_test_ script file to analyze the model's test results. Simultaneously record various evaluation metrics, such as Accuracy, F1 score, UF1 score, UAR, confusion matrix diagrams, hyperparameter optimization curves, and sensitivity curves,etc.
```
python /last_test.py
```
#### 5. Other evaluation metrics are calculated separately.
- The _resultAnalysis_ script can also compute the PR and ROC curves for test samples to further analyze model performance. These are not provided separately here; they can be directly configured and implemented within the _test_ script.


------

The algorithm is directly related to the manuscript currently submitted to Journal of Experimental & Theoretical Artificial Intelligence journal. For detailed information on the algorithm, please track subsequent manuscript updates. Currently available only for the review stage.
If you find this helpful, please cite:

    @article{Zhang2026MER,
      title={Multimodal Tangent-weighted Fusion Micro-Expression Recognition Network Using ECG-QRS Waves},
      author={Fan Zhang , Lin Chai},
      journal={Journal of Experimental & Theoretical Artificial Intelligence},
      volume={None},
      pages={None},
      year={2025-2026},
      publisher={Taylor}
    }

