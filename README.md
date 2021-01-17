# DeepAR Recurrent Neural Network Time Series Red Light Camera Violation Forecast
Created by Frank Jiang

![Project_image](https://www.cloudmanagementinsider.com/wp-content/uploads/2020/10/Amit-Cards-04-1024x535.png)

 ---
 ### Table of Contents
   - [Description](#Description)
   - [Data](#Data)
   - [Environment](#Environment)
   - [Algorithm](#Algorithm)
   - [Training](#Training)
   - [Host](#Host)
   - [Evaluation and Visualization](#Evaluation-and-visualization)
   - [Reference](#Reference)
 ---
 
 ## Description
 
 ### Background and Interest
 Time Series Forecasting has always been the major interest both in the business and social science research sector. Classic forecasting methods such as Autoregressive intergrated moving average (ARIMA) or exponential smoothing(ETS). However, in many datasets, you might have many similar time series across cross-sectional units. In this case, DeepAR algorithm provided by AWS outperforms traditional approaches. We will discuss about the algorithm in the later section. 
 
 ## Data
 
 ### Data Source
 
The dataset being used in this project are from [City of Chicago Red Light camera Violations](https://data.cityofchicago.org/Transportation/Red-Light-Camera-Violations/spqx-js37). This data is currently hosted by [Chicago Data Portal](http://www.cityofchicago.org/city/en/depts/cdot/supp_info/red-light_cameraenforcement.html). This dataset records the daily volume of violations created by the City of Chicago Red Light Program for each camera. 
 
 ### Preprocessing
 
 The dataset itself are mostly cleaned. However, there are 2 issues that needs to be processed before training. First, the duplicate index exist in the address column. Because there might be several red light cameras at the same cross road, we aggregate the address with the camera id as the index to avoid the problem of duplicate index. 
 Also, DeepAR model takes the input of a json format file with two columns, 'start'(time index) and target variable. Therefore, we encode our processed csv file into json format file. We use the following encoding and decoding method for csv to json file conversion in Sagemaker. 
  ```python
def series_2_obj(series, cat=None):
    obj = {'start': str(series.index[0]), 'target': list(series)}
    if cat:
        obj['cat'] = cat
    return obj

def series_2_json(series, cat=None):
    return json.dumps(series_2_obj(series, cat))
    
    
encoding = 'utf-8'
s3filesystem = s3fs.S3FileSystem()
with s3filesystem.open(train_data_path, 'wb') as fp:
    for series in training:
        fp.write(series_2_json(series).encode(encoding))
        fp.write('\n'.encode(encoding))

with s3filesystem.open(test_data_path, 'wb') as fp:
    for series in violation_list:
        fp.write(series_2_json(series).encode(encoding))
        fp.write('\n'.encode(encoding))
```

## Environment

This Deep learning model pipeline, training, and hosting are all conducted in [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html). Amazon SageMaker is a fully managed machine learning service that provides model building and directly deploy them into a production-ready hosted environment. All model pipeline and training are written in Python. 

To set up the environment, we created and tested on an ml.m4.xlarge instance. We first create the S3 bucket and prefix for training and model data, the IAM role arn used to give training and hosting access to your data. For initialization, please see the following code

 ```python
import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()
prefix = 'sagemaker/deepar-redlight-violation'
region = sess.boto_region_name

role = sagemaker.get_execution_role()
```

## Algorithm

DeepAR algorithm has its foundation on recurrent neural network (RNN), a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. This Algorithm developed by Amazon tailors a LSTM-based RNN architecture that implements a series of gates in which information is either passed or forgotten. Instead of just passing on a hidden state h, it also passes on a long term state c. DeepAR model allows simultaneous training of many related time-series and implements an encoder-decoder setup common in sequence-to-sequence models. This algorithm first training an encoder network on the whole data range, then outputting an initial state h. This state is then used to transfer information about the range to the prediction range through a decoder network. 

[Algorithm Explaination](https://miro.medium.com/max/3600/0*9xUS6MMxz3hCz2f7.png)


## Training

The training job in this project are conducted through an estimator object with Amazon SageMaker Python SDK. Managed spot training are being used in this model with the train_use_spot parameter. We also set the limit to at most 1 hour. 

We also uses automatic model tunning for discovering the best values for the DeepAR hyperparameter. The automatic model tunning check the context length(the number of time points that model gets before making the prediction), epochs(the maximum number of passes over the training data), num cells(the number of cells to use in each hidden layer of the RNN), learning rate, likelihood (noise model). Our model will try to minimize the root mean square error for model tuning and selection. 


## Host

The best model selected from the hyperparameter tunning are saved in a customized end-point in the S3 bucket that could be used for further prediction. 

```Python
deepar_best_model = Tunning.best_training_job()
endpoint_name = Tunning.deploy(initial_instance_count=1,
                             endpoint_name=deepar_best_model,
                             instance_type='ml.m4.xlarge',
                             wait=True)
```

## Evaluation-and-visualization

We have mentioned previously that one of the major reasons to use DeepAR is the multiple time series in this dataset. Therefore, we graph the initial visualization of the time series of the red light camera violations. 

[Initial Time Series Visualization]()

After model training, we implement a DeepAR predictor to save the prediction result in Panda dataframe instead of JSON format. We can look at the forecast including the confidence interval of the violation at each individual address(camera). For more visualization and prediction, feel free to check the python script. 

[Prediction Forecast 1]()
[Prediction Forecast 2]()
[Prediction Forecast 3]()


## Reference

For more information regarding DeepAR algorithm, SageMaker, please check the following link. 
https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html
https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html
https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_applying_machine_learning/deepar_chicago_traffic_violations/deepar_chicago_traffic_violations.ipynb
https://towardsdatascience.com/prophet-vs-deepar-forecasting-food-demand-2fdebfb8d282
