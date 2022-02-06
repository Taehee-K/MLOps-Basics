# MLOps Basics

Curriculum based on [graviraja/MLOps-Basics](https://github.com/graviraja/MLOps-Basics)

<table>
<tr>
    <td align="center"><b>Overview</b></td>
    <td align="center"><b>Weekly Curriculum</b></td>
</tr>
<tr><td>

<img width = 560 src = "https://user-images.githubusercontent.com/63901494/149688252-d8c246ea-b11d-4c0a-9f0b-69a8348bb72c.png">

</td><td>

|  #  |                 Course                 |       Status       |
| :-: | :------------------------------------: | :----------------: |
|  0  |             Project Setup              | :heavy_check_mark: |
|  1  | Model Monitoring<br>Weights and Biases | :heavy_check_mark: |
|  2  |        Configurations<br>Hydra         | :heavy_check_mark: |
|  3  |      Data Version Control<br>DVC       | :heavy_check_mark: |
|  4  |        Model Packaging<br>ONNX         | :heavy_check_mark: |
|  5  |       Model Packaging<br>Docker        |                    |
|  6  |        CI/CD<br>GitHub Actions         |                    |
|  7  |     Container Registry<br>AWS ECR      |                    |
|  8  |  Serverless Deployment<br>AWS Lambda   |                    |
|  9  |    Prediction Monitoring<br>Kibana     |                    |

</td></tr>
</table>

## Usage

### Installation

This project uses python 3.8

```
git clone https://github.com/Taehee-K/MLOps-Basics.git
cd MLOps-Basics
pip install -r requirements.txt
```

### Train

```
python train.py
```

Use `--multirun`(`-m`) to train with different parameter combinations

```
python train.py -m training.max_epochs=2,5 processing.batch_size=32,64,128
```

### Monitoring

WanDB Login

```
wandb login
```

After the training is complete, follow the link in the log to see all the plots on wandb dashboard

### Versioning Data

Install & Initialize DVC

```
pip install 'dvc[gdrive]'
dvc init
```

Configure `Google Drive` as remote storage

```
dvc remote add -d storage gdrive://{google-drive folder id}
```

Add trained model to remote storage

```
cd dvcfiles # create folder to save dvc files
dvc add {best-model-checkpoint}.ckpt --file {trained_model}.dvc
dvc push {trained_model}.dvc
```

Pull checkpoint from remote storage

```
cd dvcfiles
dvc pull {trained_model}.dvc
```

### Versioning Models

Tag the commit(version) to a particular dvc file

```
git tag -a "v{0.0}" -m "Version {0.0}"
git push origin v{0.0}
```

### Exporting model to ONNX

Convert trained model to onnx

```
python convert_model_to_onnx.py
```

### Inference

#### Inference using standard pytorch

```
python inference.py
```

#### Inference using ONNX Runtime

```
python inference_onnx.py
```

<!--
## Structure
-->
