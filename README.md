# MLOps Basics

Curriculum based on [graviraja/MLOps-Basics](https://github.com/graviraja/MLOps-Basics)

<table>
<tr>
    <td align="center"><b>Overview</b></td>
    <td align="center"><b>Weekly Curriculum</b></td>
</tr>
<tr><td>

<img width = 570 src = "https://user-images.githubusercontent.com/63901494/149688252-d8c246ea-b11d-4c0a-9f0b-69a8348bb72c.png">

</td><td>

|  #  |                 Course                 | :triangular_flag_on_post: |
| :-: | :------------------------------------: | :-----------------------: |
|  0  |             Project Setup              |    :heavy_check_mark:     |
|  1  | Model Monitoring<br>Weights and Biases |    :heavy_check_mark:     |
|  2  |        Configurations<br>Hydra         |    :heavy_check_mark:     |
|  3  |      Data Version Control<br>DVC       |    :heavy_check_mark:     |
|  4  |        Model Packaging<br>ONNX         |    :heavy_check_mark:     |
|  5  |       Model Packaging<br>Docker        |    :heavy_check_mark:     |
|  6  |        CI/CD<br>GitHub Actions         |    :heavy_check_mark:     |
|  7  |     Container Registry<br>AWS ECR      |                           |
|  8  |  Serverless Deployment<br>AWS Lambda   |                           |
|  9  |    Prediction Monitoring<br>Kibana     |                           |

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
### Inference using FastAPI

```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
-->

### Google Service Account

Create [GCP service account](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)

Add `credentials.json` file to working directory - created during gcp service account creation

**NOTE: Do NOT share `credentials.json` file publicly**

#### Configuring DVC

Use service account instead of actual google account using gcp service account credentials

```
dvc init
dvc remote add -d storage gdrive://{google-drive folder id}
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path {credentials}.json
```

### Docker

[Install Docker](https://docs.docker.com/engine/install/)

Build docker image

```
docker build -t {repository-name}:{tag} .
```

<!--
Delete docker image: docker rmi -f {image-id}
-->

Check weather image was built successfully

```
docker images
```

Run docker container

```
docker run -p 8000:8000 --name {container-name} {repository-name}:{tag}
```

<!--
Delete docker container
docker stop {continer-id}
docker rm {container-id}
-->

(or)

Build and run docker container

```
docker-compose up
```

<!--
## Structure
-->
