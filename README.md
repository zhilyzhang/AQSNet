# AQSNet: On the automatic quality assessment of annotated sample data for object extraction from remote sensing imagery ##[Paper Address](https://www.sciencedirect.com/science/article/pii/S0924271623001430)

AQSNet is a novel and automatic Annotation Quality aSsessment Network (AQSNet) for remote sensing data, where the goal is to assess the quality of annotated remote sensing samples (e.g., building, waterbody, and other object class datasets). This paper’s main contributions include three points. 1) To the best of our knowledge, we are the first to propose a fully automated method for the AQS of RS datasets. 2) A massive RS dataset, HBD4, and its manually constructed WAQS dataset are made publicly available to the research on objection extraction, land cover classification, and automatic quality assessment of RS annotated samples. 3) Our proposed method has been successfully applied in two application scenarios. On the one hand, we present a new strategy for generating simulated samples, which yields promising accuracy while significantly reducing the use of real-world training samples. On the other hand, our proposed method provides high-quality and massive samples automatically for related RS applications, such as building and water extraction.

## Downloading datasets 
*  Hubei Land Cover Satellite DataSet: [HBD4](http://58.48.42.237/luojiaSet/datasets/datasetDetail/77?id=77&taskType=lc)

HBD4 is a big remote sensing dataset, with 1,923,346 tiles of 512 × 512 size. The dataset has an image spatial resolution of 2 m and contains four bands including red, green, blue and near infrared.
<p align="center">
    <img src="figures/Hubei_Province.png" width=800></br>
    Waterbody Resource Distribution in Hubei Province.
    <img src="figures/HBD4_waterbody_samples.bmp" width=800></br>
    Some typical waterbody samples from HBD4.
</p>

*  Building Annotation Quality aSsessment Dataset: BAQS Dataset [Baidu cloud link](https://pan.baidu.com/s/1Wd-tO67Y8S7gcVWZRyWdng)(Extraction code:EVLa) 
<p align="center">
    <img src="figures/basq_samples.bmp" width=800></br>
    Visualizations for typical samples with similar IP/CP and different PoP in the BAQS dataset.
</p>

*  Waterbody Annotation Quality aSsessment Dataset: WAQS Dataset [Baidu cloud link](https://pan.baidu.com/s/1jOvkzznmtjkS8mmnBfhL7A)(Extraction code:EVLa)
<p align="center">
    <img src="figures/wasp_samples.bmp" width=800></br>
    Visualizations for typical samples with similar IP/CP and different PoPs on the WAQS dataset.
</p>


## Usage

### 1. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 2. Prepare data

After downloading the two sets of AQS datasets (BAQS and WAQS), you can use the script dataset/txt_file_prepare.py to create the necessary training txt files.
The generated examples could be like:  
data/reality_format/train.txt  
data/reality_format/test.txt

Once this is done, you can proceed with further training, prediction, and applications.

### 3. Training / Prediction / Applications

- To run the training script on the BAQS and WAQS datasets with a batch size of 8 and other specified parameters:  
  Set local_rank to specify the GPU if you are doing single-card training.  
  Set distributed to False.  
  Set with_simulated to True if using simulated samples; otherwise, set it to False.  
  Specify the pre-trained weights using pretrained_weights.  
  Specify the training sample txt file using train_txt_file.  
  Here's a command-line example for single-card training:
  ```bash
    python Scripts/train.py
  ```
  
  Using multi-card training, set local_rank to -1 and distributed to True.  
  Keep the rest of the parameters the same as before.
  Here's the command-line example for multi-card training:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 29503 --nproc_per_node=2 --nnodes=1 ./Scripts/train.py
  ```
  
- Run prediction script
  for calculating accuracy metric.
```bash
python Scripts/predict.py Experiments/aqs_metrics/water_reality.json
```

- Run application script
  to assess the quality of annotated remote sensing samples.
```bash
python Scripts/application.py Experiments/aqs_applications/building_data_checking.json
```

## Citations

```bibtex
@article{zhang2023automatic,
  title={On the automatic quality assessment of annotated sample data for object extraction from remote sensing imagery},
  author={Zhang, Zhili and Zhang, Qi and Hu, Xiangyun and Zhang, Mi and Zhu, Dehui},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={201},
  pages={153--173},
  year={2023},
  publisher={Elsevier}
}
```
