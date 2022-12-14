# EEG_Signals_Classification
**Feature extraction** of EEG signals and implementation of the best classification method (with different machine learning models like **KNN, SVM,** and **MLP**) to find the time step in which the brain realizes the concept of the picture it has been shown (the neurons activate).

General form of an **EEG** signal:

<img src="4.png" width="600" height="400">

The experiment was done with **28 electrodes** put on different regions of a person's brain, and it was repeated for multiple trials. The first step to take with signals like this, is to remove the noise and filter the signal. Then, the features corresponding to each of the trials must be extracted, and then the signal processing starts. 

In this project, it is assumed that the preprocessing phase (consisting of filtering the data, and removing the noise) has already been done. The dataset is given in 2 .mat files. The first dataset is for when a picture of a **human being** is shown to the person, and the second dataset is when the person is looking at a picture of a **piano**.

Each data is consisted of **3500 time steps**. Considering the fact that the **sampling rate** is **500 Hz**, each trial is consisted of **7 total seconds**.

The final goal of this project is to find the time step in which the brain neurons activate. 

First, we load the 2 datasets for human and piano pictures, named S2TB1, and S2TB2.

<h2> &nbsp;Feature Extraction</h2>

Here are some general explanations regarding **feature extraction** from the data:

A variety of methods have been widely used to extract the features from EEG signals. Among these methods are **Time Frequency Distributions (TFD)**, **Fast Fourier Transform (FFT)**, **Eigenvector Methods (EM)**, **Wavelet Transform (WT)**, and **Auto Regressive Method (ARM)**.

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

The samples of the k_th dataframe vary from 50k to 50k+70, so the middle sample index is (50k+50k+70)/2=50k+35.

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

