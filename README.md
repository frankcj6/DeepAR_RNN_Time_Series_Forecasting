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
   - [Resources](#Resources)
 ---
 
 ## Description
 
 ### Background and Interest
 Time Series Forecasting has always been the major interest both in the business and social science research sector. Classic forecasting methods such as Autoregressive intergrated moving average (ARIMA) or exponential smoothing(ETS). However, in many datasets, you might have many similar time series across cross-sectional units. In this case, DeepAR algorithm provided by AWS outperforms traditional approaches. We will discuss about the algorithm in the later section. 
 
 ## Data
 
 ### Data Source
 
The dataset being used in this project are from [city of Chicago Red Light camera Violations](https://data.cityofchicago.org/Transportation/Red-Light-Camera-Violations/spqx-js37). This data is currently hosted by [Chicago Data Portal](http://www.cityofchicago.org/city/en/depts/cdot/supp_info/red-light_cameraenforcement.html). This dataset records the daily volume of violations created by the City of Chicago Red Light Program for each camera. 
 
 ### Preprocessing
 
 The dataset itself are mostly cleaned. However, there are 2 issues that needs to be processed before training. First, the duplicate index exist in the address column. Because there might be several red light cameras at the same cross road, we aggregate the address with the camera id as the index to avoid the problem of duplicate index. 
 Also, DeepAR model takes the input of a json format file with two columns, 'start'(time index) and target variable. Therefore, we encode our processed csv file into json format file. We use the following encoding and decoding method for csv to json file conversion in Sagemaker. 
  ```console
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

 
 
