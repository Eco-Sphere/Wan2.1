---
pipeline_tag: text-to-video
frameworks:
  - PyTorch
license: apache-2.0
library_name: openmind
hardwares:
  - NPU
  - Atlas 800I A2
  - Atlas 800T A2
language:
  - en
---
## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.11.10 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I/800T A2(8*64G)推理设备：支持的卡数最小为1
- [Atlas 800I/800T A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 1.3 环境依赖安装
```shell
pip3 install -r requirements.txt
```

### 1.4 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.5 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

### 1.6 gcc、g++安装
```shell
# 若环境镜像中没有gcc、g++，请用户自行安装
yum install gcc
yum install g++

# 导入头文件路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
```
注：若使用openeuler镜像，需要配置gcc、g++环境，否则会导致`fatal error: 'stdio.h' file not found`

## 二、下载权重

### 2.1 权重及配置文件说明
1. Wan2.1-T2V-1.3B权重链接:
```shell
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
```
2. Wan2.1-T2V-14B权重链接
```shell
https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
```
3. Wan2.1-I2V-480P权重链接:
```shell
https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
```
4. Wan2.1-I2V-720P权重链接
```shell
https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
```

## 三、Wan2.1使用

### 3.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/Wan2.1.git
```

### 3.2 Wan2.1-T2V-1.3B

使用上一步下载的权重
```shell
model_base="./Wan2.1-T2V-1.3B/"
```
#### 3.2.1 单卡性能测试
##### 3.2.1.1 等价优化
执行命令：
```shell
# Wan2.1-T2V-1.3B
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

python generate.py  \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ${model_base} \
--sample_steps 50 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."\
--base_seed 0 
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- task: 任务类型。
- ckpt_dir: 模型的权重路径
- size: 生成视频的高和宽
- prompt: 文本提示词
- base_seed: 随机种子

##### 3.2.2.1 算法优化
执行命令：
```shell
# Wan2.1-T2V-1.3B
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

python generate.py  \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ${model_base} \
--sample_steps 50 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."\
--base_seed 0 \
--use_attentioncache \
--start_step 20 \
--attentioncache_interval 2 \
--end_step 47
```
参数说明：
- use_attentioncache: 使能attentioncache策略
- start_step: cache开始的step
- attentioncache_interval: 连续cache数
- end_step: cache结束的step

#### 3.2.2 多卡性能测试
##### 3.2.2.1 等价优化
执行命令：
```shell
# 1.3B支持双卡、四卡
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=4 generate.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ${model_base} \
--ulysses_size 4 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 0 
```

参数说明：
- dit_fsdp: DiT使用FSDP
- t5_fsdp: T5使用FSDP
- ulysses_size: ulysses并行数

##### 3.2.2.2 算法优化
执行命令：
```shell
# 1.3B支持双卡、四卡
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=4 generate.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ${model_base} \
--ulysses_size 4 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 0 \
--use_attentioncache \
--start_step 20 \
--attentioncache_interval 2 \
--end_step 47
```

参数说明：
- use_attentioncache: 使能attentioncache策略
- start_step: cache开始的step
- attentioncache_interval: 连续cache数
- end_step: cache结束的step


### 3.3 Wan2.1-T2V-14B
使用上一步下载的权重
```shell
model_base="./Wan2.1-T2V-14B/"
```
#### 3.3.1 等价优化
#### 3.3.1.1 8卡性能测试
执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=8 generate.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir ${model_base} \
--dit_fsdp \
--t5_fsdp \
--ulysses_size 8 \
--vae_parallel \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 0 
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略
- base_seed: 随机种子

#### 3.3.1.2 8卡TP性能测试
执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=8 generate.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir ${model_base} \
--t5_fsdp \
--tp_size 8 \
--vae_parallel \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 0 
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- tp_size: tensor parallel并行数
- vae_parallel: 使能vae并行策略
- base_seed: 随机种子

#### 3.3.1.3 16卡性能测试
执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=16 generate.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir ${model_base} \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 8 \
--vae_parallel \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 0
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- cfg_size: cfg并行数
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略

#### 3.3.2 算法优化
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 generate.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir ${model_base} \
--dit_fsdp \
--t5_fsdp \
--sample_steps 50 \
--ulysses_size 8 \
--vae_parallel \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--use_attentioncache \
--start_step 20 \
--attentioncache_interval 2 \
--end_step 47
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略
- use_attentioncache: 使能attentioncache策略
- start_step: cache开始的step
- attentioncache_interval: 连续cache数
- end_step: cache结束的step


### 3.4 Wan2.1-I2V-14B
使用上一步下载的权重
```shell
# 生成480P的视频
model_base="./Wan2.1-I2V-14B-480P/"
# 生成720P的视频
model_base="./Wan2.1-I2V-14B-720P/"
```

#### 3.4.1 等价优化
#### 3.4.1.1 8卡性能测试

执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=8 generate.py \
--task i2v-14B \
--size 832*480 \
--ckpt_dir ${model_base} \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--vae_parallel \
--image examples/i2v_input.JPG \
--base_seed 0 \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- task: 任务类型。
- size: 生成视频的分辨率，支持[1280, 720]、[832, 480]、[720, 480]等
- ckpt_dir: 模型的权重路径
- frame_num: 生成视频的帧数
- sample_steps: 推理步数
- dit_fsdp: DiT使用FSDP
- t5_fsdp: T5使用FSDP
- cfg_size: cfg并行数
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略
- image: 用于生成视频的图片路径
- base_seed: 随机种子
- prompt: 文本提示词

#### 3.4.1.2 8卡TP性能测试

执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=8 generate.py \
--task i2v-14B \
--size 832*480 \
--ckpt_dir ${model_base} \
--frame_num 81 \
--sample_steps 40 \
--t5_fsdp \
--tp_size 8 \
--vae_parallel \
--image examples/i2v_input.JPG \
--base_seed 0 \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- ckpt_dir: 模型的权重路径
- frame_num: 生成视频的帧数
- sample_steps: 推理步数
- t5_fsdp: T5使用FSDP
- tp_size: tensor parallel并行数
- vae_parallel: 使能vae并行策略
- base_seed: 随机种子
- prompt: 文本提示词

#### 3.4.2 算法优化
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 generate.py \
--task i2v-14B \
--size 1280*720 \
--ckpt_dir ${model_base} \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0 \
--vae_parallel \
--use_attentioncache \
--start_step 12 \
--attentioncache_interval 4 \
--end_step 37
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- cfg_size: cfg并行数
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略
- use_attentioncache: 使能attentioncache策略
- start_step: cache开始的step
- attentioncache_interval: 连续cache数
- end_step: cache结束的step

注： 若出现OOM，请添加环境变量`export T5_LOAD_CPU=1`

## 四、量化功能支持
本项目新增量化功能，支持权重 8 位（w8）与激活 8 位 / 16 位（a8/a16）的量化组合，可减少模型显存占用并保持推理性能
### 4.1 安装量化工具msModelSlim
参考[官方README](https://gitee.com/ascend/msit/tree/master/msmodelslim)
1. git clone下载msit仓代码
2. 进入到msit/msmodelslim的目录 cd msit/msmodelslim；并在进入的msmodelslim目录下，运行安装脚本 bash install.sh

### 4.2 量化模型生成
通过`quantization/quant.py`脚本生成量化模型及描述文件，需基于原始模型权重进行量化。

##### 4.2.1 量化脚本参数说明
| 参数名              | 说明                                                                 | 可选值/示例                     |
|---------------------|----------------------------------------------------------------------|--------------------------------|
| --task              | 任务类型（与推理任务一致）                                           | t2v-1.3B, t2v-14B, i2v-14B    |
| --ckpt_dir          | 原始模型权重路径                                                     | ./Wan2.1-T2V-1.3B              |
| --quant_save_dir    | 量化模型保存路径（默认./quant_weights）                              | ./my_quant_weights             |
| --quant_mode        | 量化模式（权重+激活位宽）                                            | w8a8（8bit权重+8bit激活）、w8a16（8bit权重+16bit激活） |
| --is_dynamic        | 是否启用动态量化（激活参数动态计算）                                 | （默认False，加此参数表示启用） |
| --w_sym             | 是否对权重使用对称量化                                               | （默认False，加此参数表示启用） |
| --act_method        | 激活量化方法（1=min-max，2=histogram，3=auto-mixed，推荐3）          | 1/2/3（默认3）                 |
| --disable_quant_layers | 需跳过量化的层名称列表（可选）                                       | "block.0" "block.1"            |
| --device_id         | 量化使用的NPU设备ID（默认0）                                         | 0, 1                           |

##### 4.2.2 量化脚本运行示例
以T2V-14B模型为例，生成8bit权重+8bit激活的动态量化模型：
```shell
# 环境变量与推理保持一致
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TOKENIZERS_PARALLELISM=false

model_base="Wan2.1-T2V-14B/"
python quantization/quant.py \
--task t2v-14B \
--ckpt_dir ${model_path} \
--quant_mode w8a8 \
--is_dynamic \
--w_sym \
--act_method 3 \
--quant_save_dir ./quant_w8a8_dynamic \
--device_id 0
```
执行后，`quant_w8a8_dynamic`目录下会生成两个文件：
- `quant_model_description_w8a8_dynamic.json`：量化配置描述文件（包含量化位宽、层映射等元信息）
- `quant_model_weight_w8a8_dynamic.safetensors`：量化后的权重文件（采用safe tensor格式，兼容Hugging Face生态）

### 4.3 安装量化模型推理工具NNAL神经网络加速库和torch_atb
#### 4.3.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

#### 4.3.2 安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x Ascend-cann-nnal_<version>_linux-<arch>.run
# 默认路径安装:
./Ascend-cann-nnal_<version>_linux-<arch>.run --install --torch_atb
# 配置环境变量:
source ${HOME}/Ascend/nnal/atb/set_env.sh
```

### 4.4 使用量化模型推理
使用量化模型进行推理时，需在原有generate.py命令中添加`--quant_desc_path`参数，指向量化描述文件（quant_model_description_*.json）路径，该路径需要是绝对路径，其余参数与原生模型推理一致。
#### 4.4.1 单卡量化推理示例（T2V-1.3B）
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES=0 #指定0卡
export model_path="your local Wan2.1-T2V-1.3B model path"
export file_absolute_path="your local quant description file absolute path"

python generate.py  \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ${model_path} \
  --sample_steps 50 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 0 \
  --quant_desc_path ${file_absolute_path}
```

#### 4.4.2 多卡量化推理示例（T2V-14B 8 卡）
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export model_path="your local Wan2.1-T2V-14B model path"
export file_absolute_path="your local quant description file absolute path"

torchrun --nproc_per_node=8 --master-port 29501 generate.py \
    --task t2v-14B \
    --ckpt_dir ${model_path} \
    --size 1280*720 \
    --sample_steps 50 \
    --frame_num 81 \
    --t5_fsdp \
    --ulysses_size 8 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --base_seed 0 \
    --quant_desc_path ${file_absolute_path}
```

#### 4.4.3 量化推理注意事项
- 多卡兼容性：目前量化模型在多卡推理时不支持`--dit_fsdp`参数，请避免使用该参数，否则会导致加载量化模型权重失败。
- 路径要求：`--quant_desc_path`需指向完整的量化描述文件路径（即quant_model_description_*.json），且该路径要求填写绝对路径。量化权重文件（.safetensors）需与描述文件在同一目录下，否则会提示权重加载失败。
- 任务兼容性：量化功能支持所有任务类型（T2V-1.3B、T2V-14B、I2V-14B），使用方法与上述示例一致。


## 五、常见问题
1. 若出现OOM, 可添加环境变量 `export T5_LOAD_CPU=1`，以降低显存占用
2. 若遇到报错: `Directory operation failed. Reason: Directory [/usr/local/Ascend/mindie/latest/mindie-rt/aoe] does not exist`,请设置环境变量`unset TUNE_BANK_PATH`
3. 若使用openeuler镜像, 若没有配置gcc、g++环境，会遇到报错：`fatal error: 'stdio.h' file not found`，请参考`1.6 gcc、g++安装`
4. 若循环跑纯模型推理，可能会因为HCCL端口未及时释放，导致因端口被占用而推理失败，报错：`Failed to bind the IP port. Reason: The IP address and port have been bound already.`
  `HCCL function error :HcclGetRootInfo(&hcclID), error code is 7`:  请配置`export HCCL_HOST_SOCKET_PORT_RANGE="auto"`不指定端口
  `HCCL function error :HcclGetRootInfo(&hcclID), error code is 11`: 请配置`sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015`预留端口

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。