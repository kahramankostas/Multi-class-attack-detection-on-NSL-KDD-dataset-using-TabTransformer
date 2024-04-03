```python
import pandas as pd
```

# DATA PREPROCESSING

## attacks and types included in the dataset.


```python
attack_types={"back":"dos",
"buffer_overflow":"u2r",
"ftp_write":"r2l",
"guess_passwd":"r2l",
"imap":"r2l",
"ipsweep":"probe",
"land":"dos",
"loadmodule":"u2r",
"multihop":"r2l",
"neptune":"dos",
"nmap":"probe",
"perl":"u2r",
"phf":"r2l",
"pod":"dos",
"portsweep":"probe",
"rootkit":"u2r",
"satan":"probe",
"smurf":"dos",
"spy":"r2l",
"teardrop":"dos",
"warezclient":"r2l",
"warezmaster":"r2l",
"normal":"normal",
"unknown":"unknown"}

```


```python
attack_types={'back': 'dos', 'buffer_overflow': 'u2r', 'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'ipsweep': 'probe', 'land': 'dos', 'loadmodule': 'u2r', 'multihop': 'r2l', 'neptune': 'dos', 'nmap': 'probe', 'normal': 'normal', 'perl': 'u2r', 'phf': 'r2l', 'pod': 'dos', 'portsweep': 'probe', 'rootkit': 'u2r', 'satan': 'probe', 'smurf': 'dos', 'spy': 'r2l', 'teardrop': 'dos', 'warezclient': 'r2l', 'warezmaster': 'r2l', 'unknown': 'unknown', 'apache2': 'dos', 'httptunnel': 'u2r', 'mailbomb': 'dos', 'mscan': 'probe', 'named': 'r2l', 'processtable': 'dos', 'ps': 'u2r', 'saint': 'probe', 'sendmail': 'r2l', 
 'snmpgetattack': 'dos', 'snmpguess': 'r2l', 'sqlattack': 'u2r', 'udpstorm': 'dos', 'worm': 'dos', 'xlock': 'r2l', 'xsnoop': 'r2l', 'xterm': 'u2r'}
```


```python
# c_names --->  column names
c_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","difficulty_degree"]

train = pd.read_csv( "data/KDDTrain+.csv", names=c_names) # train file
test = pd.read_csv("data/KDDTest+.csv", names=c_names) # test file
```


```python

```

# Attack Distrubution


```python
# Attack Distrubution
print("#####################  TRAIN  #############################################")
print(train.groupby("labels").size())
print("\n\n\n\n#####################  TEST  #############################################")
print(test.groupby("labels").size())
```

    #####################  TRAIN  #############################################
    labels
    back                 956
    buffer_overflow       30
    ftp_write              8
    guess_passwd          53
    imap                  11
    ipsweep             3599
    land                  18
    loadmodule             9
    multihop               7
    neptune            41214
    nmap                1493
    normal             67343
    perl                   3
    phf                    4
    pod                  201
    portsweep           2931
    rootkit               10
    satan               3633
    smurf               2646
    spy                    2
    teardrop             892
    warezclient          890
    warezmaster           20
    dtype: int64
    
    
    
    
    #####################  TEST  #############################################
    labels
    apache2             737
    back                359
    buffer_overflow      20
    ftp_write             3
    guess_passwd       1231
    httptunnel          133
    imap                  1
    ipsweep             141
    land                  7
    loadmodule            2
    mailbomb            293
    mscan               996
    multihop             18
    named                17
    neptune            4657
    nmap                 73
    normal             9710
    perl                  2
    phf                   2
    pod                  41
    portsweep           157
    processtable        685
    ps                   15
    rootkit              13
    saint               319
    satan               735
    sendmail             14
    smurf               665
    snmpgetattack       178
    snmpguess           331
    sqlattack             2
    teardrop             12
    udpstorm              2
    warezmaster         944
    worm                  2
    xlock                 9
    xsnoop                4
    xterm                13
    dtype: int64
    


```python
# removing of unnecessary feature (difficulty_degree)
del train["difficulty_degree"] 
del test["difficulty_degree"] 
```


```python
train["labels"]=train["labels"].replace(attack_types)
test["labels"]=test["labels"].replace(attack_types)
```


```python
# NEW grouped Attack Distrubution
print("#####################  TRAIN  #############################################")
print(train.groupby("labels").size())
print("\n\n#####################  TEST  #############################################")
print(test.groupby("labels").size())
```

    #####################  TRAIN  #############################################
    labels
    dos       45927
    normal    67343
    probe     11656
    r2l         995
    u2r          52
    dtype: int64
    
    
    #####################  TEST  #############################################
    labels
    dos       7638
    normal    9710
    probe     2421
    r2l       2574
    u2r        200
    dtype: int64
    

# Multi-class attack detection using TabTransformer

The binary classification code in this source has been refactored for multiclass implementation:
https://keras.io/examples/structured_data/tabtransformer/


```python
import keras
from keras import layers
from keras import ops

import math
import numpy as np
import pandas as pd
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from functools import partial
```


```python
CSV_HEADER = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]


feat = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","difficulty_degree"]


train_data_url = "data/KDDTrain+.csv"

train_data = pd.read_csv(train_data_url, header=None, names=feat)

test_data_url = "data/KDDTest+.csv"
test_data = pd.read_csv(test_data_url, header=None, names=feat)

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")

"""
Remove the first record (because it is not a valid data example) and a trailing 'dot' in the class labels.
"""

"""
Now we store the training and test data in separate CSV files.
"""

train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

# removing of unnecessary feature (difficulty_degree)
del train_data["difficulty_degree"] 
del test_data["difficulty_degree"] 

train_data["labels"]=train_data["labels"].replace(attack_types)
test_data["labels"]=test_data["labels"].replace(attack_types)


train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)

```

    Train dataset shape: (125973, 43)
    Test dataset shape: (22543, 43)
    


```python
train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>protocol_type</th>
      <th>service</th>
      <th>flag</th>
      <th>src_bytes</th>
      <th>dst_bytes</th>
      <th>land</th>
      <th>wrong_fragment</th>
      <th>urgent</th>
      <th>hot</th>
      <th>...</th>
      <th>dst_host_srv_count</th>
      <th>dst_host_same_srv_rate</th>
      <th>dst_host_diff_srv_rate</th>
      <th>dst_host_same_src_port_rate</th>
      <th>dst_host_srv_diff_host_rate</th>
      <th>dst_host_serror_rate</th>
      <th>dst_host_srv_serror_rate</th>
      <th>dst_host_rerror_rate</th>
      <th>dst_host_srv_rerror_rate</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>tcp</td>
      <td>ftp_data</td>
      <td>SF</td>
      <td>491</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>25</td>
      <td>0.17</td>
      <td>0.03</td>
      <td>0.17</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>udp</td>
      <td>other</td>
      <td>SF</td>
      <td>146</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.60</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>tcp</td>
      <td>private</td>
      <td>S0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>26</td>
      <td>0.10</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>dos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>tcp</td>
      <td>http</td>
      <td>SF</td>
      <td>232</td>
      <td>8153</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>255</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>tcp</td>
      <td>http</td>
      <td>SF</td>
      <td>199</td>
      <td>420</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>255</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>125968</th>
      <td>0</td>
      <td>tcp</td>
      <td>private</td>
      <td>S0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>25</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>dos</td>
    </tr>
    <tr>
      <th>125969</th>
      <td>8</td>
      <td>udp</td>
      <td>private</td>
      <td>SF</td>
      <td>105</td>
      <td>145</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>244</td>
      <td>0.96</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125970</th>
      <td>0</td>
      <td>tcp</td>
      <td>smtp</td>
      <td>SF</td>
      <td>2231</td>
      <td>384</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>30</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.72</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125971</th>
      <td>0</td>
      <td>tcp</td>
      <td>klogin</td>
      <td>S0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>dos</td>
    </tr>
    <tr>
      <th>125972</th>
      <td>0</td>
      <td>tcp</td>
      <td>ftp_data</td>
      <td>SF</td>
      <td>151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>77</td>
      <td>0.30</td>
      <td>0.03</td>
      <td>0.30</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
<p>125973 rows × 42 columns</p>
</div>




```python
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>protocol_type</th>
      <th>service</th>
      <th>flag</th>
      <th>src_bytes</th>
      <th>dst_bytes</th>
      <th>land</th>
      <th>wrong_fragment</th>
      <th>urgent</th>
      <th>hot</th>
      <th>...</th>
      <th>dst_host_srv_count</th>
      <th>dst_host_same_srv_rate</th>
      <th>dst_host_diff_srv_rate</th>
      <th>dst_host_same_src_port_rate</th>
      <th>dst_host_srv_diff_host_rate</th>
      <th>dst_host_serror_rate</th>
      <th>dst_host_srv_serror_rate</th>
      <th>dst_host_rerror_rate</th>
      <th>dst_host_srv_rerror_rate</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>tcp</td>
      <td>private</td>
      <td>REJ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>0.04</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>dos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>tcp</td>
      <td>private</td>
      <td>REJ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>dos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>tcp</td>
      <td>ftp_data</td>
      <td>SF</td>
      <td>12983</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>86</td>
      <td>0.61</td>
      <td>0.04</td>
      <td>0.61</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>icmp</td>
      <td>eco_i</td>
      <td>SF</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>57</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.28</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>probe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>tcp</td>
      <td>telnet</td>
      <td>RSTO</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>86</td>
      <td>0.31</td>
      <td>0.17</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.83</td>
      <td>0.71</td>
      <td>probe</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22538</th>
      <td>0</td>
      <td>tcp</td>
      <td>smtp</td>
      <td>SF</td>
      <td>794</td>
      <td>333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>141</td>
      <td>0.72</td>
      <td>0.06</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>22539</th>
      <td>0</td>
      <td>tcp</td>
      <td>http</td>
      <td>SF</td>
      <td>317</td>
      <td>938</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>255</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>22540</th>
      <td>0</td>
      <td>tcp</td>
      <td>http</td>
      <td>SF</td>
      <td>54540</td>
      <td>8314</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>255</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>dos</td>
    </tr>
    <tr>
      <th>22541</th>
      <td>0</td>
      <td>udp</td>
      <td>domain_u</td>
      <td>SF</td>
      <td>42</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>252</td>
      <td>0.99</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>22542</th>
      <td>0</td>
      <td>tcp</td>
      <td>sunrpc</td>
      <td>REJ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>21</td>
      <td>0.08</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.44</td>
      <td>1.00</td>
      <td>probe</td>
    </tr>
  </tbody>
</table>
<p>22543 rows × 42 columns</p>
</div>




```python

# A list of the numerical feature names.
NUMERIC_FEATURE_NAMES = ["duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]
# A dictionary of the categorical features and their vocabulary.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "protocol_type": sorted(list(train_data["protocol_type"].unique())),
    "service": sorted(list(train_data["service"].unique())),
    "flag": sorted(list(train_data["flag"].unique()))}





# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
# A list of all the input features.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# A list of column default values for each feature.
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES else ["NA"]
    for feature_name in CSV_HEADER
]
# The name of the target feature.
TARGET_FEATURE_NAME = "labels"
# A list of the labels of the target features.
TARGET_LABELS = ["dos", 
"normal",  
"probe", 
"r2l" ,   
"u2r"]
```


```python
"""
## Configure the hyperparameters

The hyperparameters includes model architecture and training configurations.
"""

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 265
NUM_EPOCHS = 15

NUM_TRANSFORMER_BLOCKS = 3  # Number of transformer blocks.
NUM_HEADS = 4  # Number of attention heads.
EMBEDDING_DIMS = 16  # Embedding dimensions of the categorical features.
MLP_HIDDEN_UNITS_FACTORS = [
    2,
    1,
]  # MLP hidden layer units, as factors of the number of inputs.
NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.

"""
## Implement data reading pipeline

We define an input function that reads and parses the file, then converts features
and labels into a[`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets)
for training or evaluation.
"""

target_label_lookup = layers.StringLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)


def prepare_example(features, target):
    target_index = target_label_lookup(target)
    return features, target_index


lookup_dict = {}
for feature_name in CATEGORICAL_FEATURE_NAMES:
    vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
    # Create a lookup to convert a string values to an integer indices.
    # Since we are not using a mask token, nor expecting any out of vocabulary
    # (oov) token, we set mask_token to None and num_oov_indices to 0.
    lookup = layers.StringLookup(
        vocabulary=vocabulary, mask_token=None, num_oov_indices=0
    )
    lookup_dict[feature_name] = lookup


def encode_categorical(batch_x, batch_y):
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        batch_x[feature_name] = lookup_dict[feature_name](batch_x[feature_name])

    return batch_x, batch_y


def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    dataset = (
        tf_data.experimental.make_csv_dataset(
            csv_file_path,
            batch_size=batch_size,
            column_names=CSV_HEADER,
            column_defaults=COLUMN_DEFAULTS,
            label_name=TARGET_FEATURE_NAME,
            num_epochs=1,
            header=False,
            na_value="?",
            shuffle=shuffle,
        )
        .map(prepare_example, num_parallel_calls=tf_data.AUTOTUNE, deterministic=False)
        .map(encode_categorical)
    )
    return dataset.cache()


```


```python
def run_experiment(
    model,
    train_data_file,
    test_data_file,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_data_file, batch_size)

    print("Start training the model...")
    history = model.fit(
        train_dataset, epochs=num_epochs, validation_data=validation_dataset
    )
    print("Model training finished")

    _, accuracy = model.evaluate(validation_dataset, verbose=0)

    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history

```


```python
def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="float32"
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="float32"
            )
    return inputs
```


```python
def encode_inputs(inputs, embedding_dims):
    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token, nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and num_oov_indices to 0.

            # Convert the string input values into integer indices.

            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(inputs[feature_name])
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:
            # Use the numerical features as-is.
            numerical_feature = ops.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list




```


```python
def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer()),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)
```

# baseline model


```python
def create_baseline_model(
    embedding_dims, num_mlp_blocks, mlp_hidden_units_factors, dropout_rate
):
    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Concatenate all features.
    features = layers.concatenate(
        encoded_categorical_feature_list + numerical_feature_list
    )
    # Compute Feedforward layer units.
    feedforward_units = [features.shape[-1]]

    # Create several feedforwad layers with skip connections.
    for layer_idx in range(num_mlp_blocks):
        features = create_mlp(
            hidden_units=feedforward_units,
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization,
            name=f"feedforward_{layer_idx}",
        )(features)

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization,
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(5, activation="softmax", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


baseline_model = create_baseline_model(
    embedding_dims=EMBEDDING_DIMS,
    num_mlp_blocks=NUM_MLP_BLOCKS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", baseline_model.count_params())
keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")

"""
Let's train and evaluate the baseline model:
"""

history = run_experiment(
    model=baseline_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)




```

    Total model weights: 47961
    Start training the model...
    Epoch 1/15
        475/Unknown [1m12s[0m 13ms/step - accuracy: 0.8468 - loss: 0.4826

    C:\Users\kahra\AppData\Local\Programs\Python\Python310\lib\contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
      self.gen.throw(typ, value, traceback)
    

    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 16ms/step - accuracy: 0.8470 - loss: 0.4819 - val_accuracy: 0.7060 - val_loss: 1.5205
    Epoch 2/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 12ms/step - accuracy: 0.9446 - loss: 0.1710 - val_accuracy: 0.7323 - val_loss: 1.5765
    Epoch 3/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 12ms/step - accuracy: 0.9654 - loss: 0.1092 - val_accuracy: 0.7333 - val_loss: 1.7363
    Epoch 4/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 16ms/step - accuracy: 0.9723 - loss: 0.0880 - val_accuracy: 0.7318 - val_loss: 1.8649
    Epoch 5/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 18ms/step - accuracy: 0.9765 - loss: 0.0757 - val_accuracy: 0.7359 - val_loss: 1.7981
    Epoch 6/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 13ms/step - accuracy: 0.9798 - loss: 0.0646 - val_accuracy: 0.7449 - val_loss: 1.9379
    Epoch 7/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 15ms/step - accuracy: 0.9837 - loss: 0.0562 - val_accuracy: 0.7433 - val_loss: 1.9181
    Epoch 8/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 17ms/step - accuracy: 0.9852 - loss: 0.0503 - val_accuracy: 0.7439 - val_loss: 1.8484
    Epoch 9/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 17ms/step - accuracy: 0.9869 - loss: 0.0460 - val_accuracy: 0.7318 - val_loss: 2.0557
    Epoch 10/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 14ms/step - accuracy: 0.9873 - loss: 0.0443 - val_accuracy: 0.7458 - val_loss: 2.0348
    Epoch 11/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 15ms/step - accuracy: 0.9884 - loss: 0.0403 - val_accuracy: 0.7445 - val_loss: 2.0181
    Epoch 12/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 13ms/step - accuracy: 0.9888 - loss: 0.0399 - val_accuracy: 0.7454 - val_loss: 2.0735
    Epoch 13/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 13ms/step - accuracy: 0.9901 - loss: 0.0363 - val_accuracy: 0.7437 - val_loss: 2.1395
    Epoch 14/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 12ms/step - accuracy: 0.9901 - loss: 0.0344 - val_accuracy: 0.7462 - val_loss: 2.1733
    Epoch 15/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 12ms/step - accuracy: 0.9897 - loss: 0.0345 - val_accuracy: 0.7438 - val_loss: 2.2905
    Model training finished
    Validation accuracy: 74.38%
    

#  Multi-class TabTransformer classifier


```python
def create_tabtransformer_classifier(
    num_transformer_blocks,
    num_heads,
    embedding_dims,
    mlp_hidden_units_factors,
    dropout_rate,
    use_column_embedding=False,
):
    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Stack categorical feature embeddings for the Tansformer.
    encoded_categorical_features = ops.stack(encoded_categorical_feature_list, axis=1)
    # Concatenate numerical features.
    numerical_features = layers.concatenate(numerical_feature_list)

    # Add column embedding to categorical feature embeddings.
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=embedding_dims
        )
        column_indices = ops.arange(start=0, stop=num_columns, step=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(
            column_indices
        )

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)
        # Skip connection 1.
        x = layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, encoded_categorical_features]
        )
        # Layer normalization 1.
        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
        # Feedforward.
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=partial(
                layers.LayerNormalization, epsilon=1e-6
            ),  # using partial to provide keyword arguments before initialization
            name=f"feedforward_{block_idx}",
        )(x)
        # Skip connection 2.
        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        # Layer normalization 2.
        encoded_categorical_features = layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Flatten the "contextualized" embeddings of the categorical features.
    categorical_features = layers.Flatten()(encoded_categorical_features)
    # Apply layer normalization to the numerical features.
    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
    # Prepare the input for the final MLP block.
    features = layers.concatenate([categorical_features, numerical_features])

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization,
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(5, activation="softmax", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


tabtransformer_model = create_tabtransformer_classifier(
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS,
    embedding_dims=EMBEDDING_DIMS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", tabtransformer_model.count_params())
keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir="LR")

"""
Let's train and evaluate the TabTransformer model:
"""

history = run_experiment(
    model=tabtransformer_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)
```

    Total model weights: 46745
    Start training the model...
    Epoch 1/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 57ms/step - accuracy: 0.9139 - loss: 0.2757 - val_accuracy: 0.7405 - val_loss: 1.5300
    Epoch 2/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 59ms/step - accuracy: 0.9796 - loss: 0.0679 - val_accuracy: 0.7510 - val_loss: 1.7431
    Epoch 3/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 56ms/step - accuracy: 0.9851 - loss: 0.0480 - val_accuracy: 0.7474 - val_loss: 1.8689
    Epoch 4/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 52ms/step - accuracy: 0.9876 - loss: 0.0403 - val_accuracy: 0.7524 - val_loss: 1.9049
    Epoch 5/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 53ms/step - accuracy: 0.9892 - loss: 0.0352 - val_accuracy: 0.7470 - val_loss: 2.0043
    Epoch 6/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 52ms/step - accuracy: 0.9904 - loss: 0.0318 - val_accuracy: 0.7498 - val_loss: 2.0176
    Epoch 7/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 47ms/step - accuracy: 0.9918 - loss: 0.0284 - val_accuracy: 0.7472 - val_loss: 2.0538
    Epoch 8/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 52ms/step - accuracy: 0.9919 - loss: 0.0277 - val_accuracy: 0.7535 - val_loss: 1.9493
    Epoch 9/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 51ms/step - accuracy: 0.9925 - loss: 0.0260 - val_accuracy: 0.7427 - val_loss: 2.2448
    Epoch 10/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 51ms/step - accuracy: 0.9928 - loss: 0.0253 - val_accuracy: 0.7459 - val_loss: 2.2222
    Epoch 11/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 51ms/step - accuracy: 0.9931 - loss: 0.0241 - val_accuracy: 0.7530 - val_loss: 2.1621
    Epoch 12/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m24s[0m 51ms/step - accuracy: 0.9935 - loss: 0.0235 - val_accuracy: 0.7473 - val_loss: 2.2312
    Epoch 13/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 52ms/step - accuracy: 0.9932 - loss: 0.0235 - val_accuracy: 0.7441 - val_loss: 2.4996
    Epoch 14/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 54ms/step - accuracy: 0.9938 - loss: 0.0223 - val_accuracy: 0.7471 - val_loss: 2.3493
    Epoch 15/15
    [1m476/476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 53ms/step - accuracy: 0.9940 - loss: 0.0225 - val_accuracy: 0.7465 - val_loss: 2.4066
    Model training finished
    Validation accuracy: 74.65%
    
