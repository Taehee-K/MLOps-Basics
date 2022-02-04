# MLOps Basics

Curriculum based on [graviraja/MLOps-Basics](https://github.com/graviraja/MLOps-Basics)

<table>
<tr>
    <td align="center">Overview</td>
    <td align="center">Weekly Curriculum</td>
</tr>
<tr><td>

<img width = 500 src = "https://user-images.githubusercontent.com/63901494/149688252-d8c246ea-b11d-4c0a-9f0b-69a8348bb72c.png">

</td><td>

| Week |                 Course                 |       Status       |
| :--: | :------------------------------------: | :----------------: |
|  0   |             Project Setup              | :heavy_check_mark: |
|  1   | Model Monitoring<br>Weights and Biases | :heavy_check_mark: |
|  2   |        Configurations<br>Hydra         | :heavy_check_mark: |

<!--
|  3   | Project Setup | :heavy_check_mark: |
|  4   | Project Setup | :heavy_check_mark: |
|  5   | Project Setup | :heavy_check_mark: |
|  6   | Project Setup | :heavy_check_mark: |
|  7   | Project Setup | :heavy_check_mark: |
|  8   | Project Setup | :heavy_check_mark: |
|  9   | Project Setup | :heavy_check_mark: |
-->
</td></tr>
</table>

## Usage

### Installation

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

<!--

Visualize TensorBoard Logs

```
tensorboard --logdir logs/cola
```
-->

WanDB Login

```
wandb login
```

After the training is complete, follow the link in the log to see all the plots on wandb dashboard

```
wandb: Synced 5 W&B file(s), 6 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Synced bert: https://wandb.ai/taehee-k/ops-basics/runs/3qfxb36f
```

### Inference

```
python inference.py
```
