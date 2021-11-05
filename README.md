

```{r, engine='bash', count_lines}
 ________  _____      ______   _____  
|_   __  ||_   _|   .' ____ \ |_   _| 
  | |_ \_|  | |     | (___ \_|  | |   
  |  _| _   | |   _  _.____`.   | |   
 _| |__/ | _| |__/ || \____) | _| |_  
|________||________| \______.'|_____| 
    
```
<!-- https://patorjk.com/software/taag/#p=testall&f=3D-ASCII&t=ELSI -->


##  How to use

### 0. Download related data sets and models

#### Data Sets: https://drive.google.com/drive/folders/1d4VcGI5GVayqj40T2EL52bs3GOa2vcYt?usp=sharing

(Due to the limitation of Google Drive storage space, only OSM1 and Skewed are shared. If you need Uniform and OSM2, please email me.)

### 1. Required libraries

#### LibTorch
homepage: https://pytorch.org/get-started/locally/

CPU version: https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip

For GPU version, choose according to your setup.

#### boost

homepage: https://www.boost.org/

### 2. Change Makefile

Choose CPU or GPU

```
# TYPE = CPU
TYPE = GPU

Change *home/liuguanli/Documents/libtorch_gpu* to your own path.

ifeq ($(TYPE), GPU)
	INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
else
	INCLUDE = -I/home/liuguanli/Documents/libtorch/include -I/home/liuguanli/Documents/libtorch/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch/lib
endif
```

### 3. Change Exp.cpp

comment *#define use_gpu* to use CPU version

```C++
#ifndef use_gpu
#define use_gpu
.
.
.
#endif  // use_gpu
```

### 4. Change path

Change */home/research/datasets/* to the position for data sets.

Change */home/liuguanli/Dropbox/research/BASE/* to your own path for ELSI.

Change */home/liuguanli/Documents/pre_train/* to the position where your store the pre-trained models.

Change the path if you do not want to store the datasets under the project's root path.

Constants.h
```C++
const string Constants::RECORDS = "./files/records/";
const string Constants::PRE_TRAIN_1D_DATA = "/home/liuguanli/Documents/pre_train/1D_data/0.1/";
const string Constants::RL_FILE = "/home/liuguanli/Dropbox/research/BASE/method_pool/RL/rl_4_sfc/RL_4_SFC.py";
```

