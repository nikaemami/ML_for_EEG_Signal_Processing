# EEG_Signals_Classification
**Feature extraction** of EEG signals and implementation of the best classification method (with different machine learning models like **KNN, SVM,** and **MLP**) to find the time step in which the brain realizes the concept of the picture it has been shown (the neurons activate).

General form of an **EEG** signal:

<img src="4.png" width="500" height="300">

The experiment was done with **28 electrodes** put on different regions of a person's brain, and it was repeated for multiple trials. The first step to take with signals like this, is to remove the noise and filter the signal. Then, the features corresponding to each of the trials must be extracted, and then the signal processing starts. 

In this project, it is assumed that the preprocessing phase (consisting of filtering the data, and removing the noise) has already been done. The dataset is given in 2 .mat files. The first dataset is for when a picture of a **human being** is shown to the person, and the second dataset is when the person is looking at a picture of a **piano**.

Each data is consisted of **3500 time steps**. Considering the fact that the **sampling rate** is **500 Hz**, each trial is consisted of **7 total seconds**.

The final goal of this project is to find the time step in which the brain neurons activate. 

First, we load the 2 datasets for human and piano pictures, named S2TB1, and S2TB2.

<h2> &nbsp;Feature Extraction</h2>

here are some general explanations regarding feature extraction from the data:

A feature represents a distinguishing property, a recognizable measurement, and a functional component obtained from a section of a pattern.

Extracted features are meant to **minimize the loss** of important information embedded in the signal. In addition, they also simplify the amount of resources needed to describe a huge set of data accurately. This is necessary to **minimize the complexity** of implementation to reduce the cost of information processing, and to cancel the potential need to compress the information.

A variety of methods have been widely used to extract the features from EEG signals. Among these methods are **Time Frequency Distributions (TFD)**, **Fast Fourier Transform (FFT)**, **Eigenvector Methods (EM)**, **Wavelet Transform (WT)**, and **Auto Regressive Method (ARM)**.

Acclaim about the definite priority of methods according to their capability is very hard. The findings indicate that each method has specific advantages and disadvantages, which make it appropriate for special type of signals.

Frequency domain methods may not provide high-quality performance for some EEG signals. In contrast, time-frequency methods, for instance, may not provide detailed information on EEG analysis as much as frequency domain methods. It is crucial to make clear the of the signal to be analyzed in the application of the method whenever the performance of analyzing method is discussed.

In this project, we used the method of dividing the data into different **overlapping windows**, as a method for feature selection. The overlapping windows were implemented as below:

```ruby
def create_window (mat_1,num_trials,num_channels):
    windowed_data = []
    for i in range (num_trials):
        for j in range (num_channels):
            for k in range(0,3500,50):
                a = sum(mat_1['a'][i,j,k:k+70])/70
                windowed_data.append(a)
    windowed_data = np.reshape(windowed_data, (45,126,70))
    return windowed_data
```

We put the data into **pandas dataframe** formats and concate the piano and human labels into 1 dataframe. We seperated the dataframes for each of the 70 time slots that we have, so finally we have 70 dataframes with 126x90 rows each.

<h4> &nbsp;Preprocessing:</h4>

We shuffle our dataframe, split it into **test** and **train** datasets , and use standardscaler and random seed on our test and train data. standardscaler moves the mean to 0 and scales the variance to 1.

<h2> &nbsp;Classification Model:</h2>

For classification, we have 3 classification models. **KNN**, **SVM**, and **Neural Networks(MLP)**.

<h4> &nbsp;K-Nearest Neighbors(KNN):</h4>

The K-nearest neighbor (KNN) algorithm is one of the simplest and earliest classification algorithms. The KNN algorithm does not require to consider probability values. The ‘K’ is the KNN algorithm is the number of nearest neighbors considered to take ‘vote’ from. The selection of different values for ‘K’ can generate different classification results for the same sample object.

<h4> &nbsp;Support vector machine(SVM):</h4>

Support vector machine (SVM) algorithm can classify both linear and non-linear data. It first maps each data item into an n-dimensional feature space where n is the number of features. It then identifies the hyperplane that separates the data items into two classes while maximizing the marginal distance for both classes and minimizing the classification errors.

<h4> &nbsp;MLP (Neural Networks):</h4>

Neural Networks are a set of machine learning algorithms which are inspired by the functioning of the neural networks of human brain. Likewise, NN algorithms can be represented as an interconnected group of nodes. The output of one node goes as input to another node for subsequent processing according to the interconnection. Nodes are normally grouped into a matrix called layer depending on the transformation they perform. Nodes and edges have weights that enable to adjust signal strengths of communication which can be amplified or weakened through repeated training. Based on the training and subsequent adaption of the matrices, node and edge weights, NNs can make a prediction for the test data.


<h2> &nbsp;Cross Validation:</h2>

We implement the 3 models discussed above in different functions. We use **GridSearchCV** that is a type of **K-fold cross validation** method to find the hyperparameters. After finding the best hyperparameters, we trained our model, evaluated it on test data, and returned the **test accuracy**.

The functions mentioned above are named as below in the main code:

```ruby
def classification_svm(df,seed_num)
def classification_mlp(df,seed_num)
def classification_knn(df,seed_num)
```
Each of them uses the corresponding function in the sklearn library:

```ruby
clf_svm = SVC(kernel='rbf',gamma='auto')
clf_mlp = MLPClassifier( learning_rate='adaptive',max_iter=1000,activation='relu',solver='adam',hidden_layer_sizes=375)
clf_knn = KNeighborsClassifier( weights='distance',algorithm='auto',metric='manhattan')
```

We defined a function called "Repeated_Classification". It classifies a dataframe 5 times with the selected model. To achive better accuracy, we use 5 different random seed numbers to classify the dataframe 5 times. For each dataframe, we get the mean and the standard deviation of the accuracies. We return a list of accuracy means and a list of accuracy stds:

```ruby
def Repeated_Classification(df,classifier_nam)
```

We define a function called "plot_accuracy". It gets the list of **accuracy_mean** and **accuracy_stds**, plots the classification accuracy during time, and reports the best accuracy.

We plot the mean accuracy and the standard error band: SE=(std)/(N^0.5)

N: number of samples

std: sample standard deviation

SE:standard error of the sample

So the maximum value of the band is mean+SE and the minimum value of the band is mean-SE.

We also mapped the number of our dataframes in time. For each dataframe, we flattend the mean of the window with the length of 70 samples. So for each dataframe, we converted the number of its middle samples to milliseconds:

the samples of the k_th dataframe vary from 50k to 50k+70, so the middle sample index is (50k+50k+70)/2=50k+35.

```ruby
accuracy_mean_list_percent=np.array(accuracy_mean_list)*100
accuracy_std_list_percent=(np.array(accuracy_std_list)/(5**0.5))*100
t = np.arange(len(accuracy_mean_list_percent))*100+70
```

<h4> MLP Results:</h4>

![My Image](1.png)

As we can see at t = 4.447 ms we got the maximum accuracy, which means in that time the brain distinguished whether the picture is a piano or a human.

<h4> SVM Results:</h4>

![My Image](2.png)

As we can see at t = 4.447 ms we got the maximum accuracy, which means in that time the brain distinguished whether the picture is a piano or a human.

<h4> KNN Results:</h4>

![My Image](3.png)

As we can see at t = 4.447 ms we got the maximum accuracy, which means in that time the brain distinguished whether the picture is a piano or a human.

So we got the same result with different accuracies for our 3 different models, and we found the time that the neurons take to distinguish a picture of a human from a picture of a piano.

