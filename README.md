# SLING and NRNS-GD/Straight-GD
This repository includes the code that can be used to reproduce the results of our paper [Last-Mile Embodied Visual Navigation](https://jbwasse2.github.io/portfolio/SLING/). 

This branch contains code to run a __NRNS-GD and Straight-GD__ exploration with SLING, if you are interested in SLING + OVRL, please see the Experiment Variations section.

## 🗺 Table of Contents
<div class="toc">
<ul>
<li><a href="#-experiment-variations">🔬 Experiment Variations</a></li>
<li><a href="#-nrns-and-straight-installation"> 💿 NRNS and Straight Installation</a></li>
<li><a href="#-running-nrns-and-straight"> 🏃 Running NRNS and Straight</a></li>
<li><a href="#-citation"> 📝 Citation</a></li>
</ul>
</li>
</ul>
</div>



## 🔬 Experiment Variations
To reproduce the results from DDPPO-GD/OVRL-GD + SLING.
```
git checkout ovrl
```

## 💿 NRNS and Straight Installation
This code base directly incorporates SLING into [NRNS](https://github.com/meera1hahn/NRNS/), therefore one only needs to follow their instructions for installing and running their codebase. We have pasted and updated their instructions here. Additionally, we have an extra few steps at the end that are required for SLING.

This project is developed with Python 3.6. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment:

```bash
conda create -n nrns python=3.6
conda activate nrns
```

### Install Habitat and Other Dependencies

NRNS makes extensive use of the Habitat Simulator and Habitat-Lab developed by FAIR. You will first need to install both [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab). 

Please find the instructions to install habitat [here](https://github.com/facebookresearch/habitat-lab#installation)

If you are using conda, Habitat-Sim can easily be installed with

```bash
conda install -c aihabitat -c conda-forge habitat-sim headless
```

We recommend downloading the test scenes and running the example script as described [here](https://github.com/facebookresearch/habitat-lab/blob/v0.1.5/README.md#installation) to ensure the installation of Habitat-Sim and Habitat-Lab was successful. Now you can clone this repository and install the rest of the dependencies:

```bash
git clone git@github.com:Jbwasse2/SLING.git
cd NRNS
python -m pip install -r requirements.txt
python download_aux.py
```

### Download Scene Data

Like Habitat-Lab, we expect a `data` folder (or symlink) with a particular structure in the top-level directory of this project. Running the `download_aux.py` script will download the pretrained models but you will still need to download the scene data. We evaluate our agents on Matterport3D (MP3D) and Gibson scene reconstructions.

#### Image-Nav Test Episodes 
The image-nav test episodes used in this paper for MP3D and Gibson can be found [here.](https://meerahahn.github.io/nrns/data) These were used to test all baselines and NRNS.


#### Matterport3D

The official Matterport3D download script (`download_mp.py`) can be accessed by following the "Dataset Download" instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded this way:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract this data to `data/scene_datasets/mp3d` such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 total scenes. We follow the standard train/val/test splits. 

#### Gibson 

The official Gibson dataset can be accessed on their [project webpage](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md). Please follow the link to download the Habitat Simulator compatible data. The link will first take you to the license agreement and then to the data. We follow the standard train/val/test splits. 

#### Final instructions for SLING
We trained a new GD model for NRNS, move it into the models folder with
```
mv gibson_NRNS_ours.pt ./models/gibson/
```
## 🏃 Running NRNS and Straight

This code can be run by changing directories to './src/image_nav/' and then running one of the following commands.
### SLING + NRNS x Gibson x Curved
```
python run.py --dataset=gibson --path_type=curved --difficulty=easy --distance_model_path=NRNS_ours.pt --model TopoGCNNRNS --tag nrns_and_sling_gibson_curved --use_glue_rhophi --dont_reuse_poses --switch_threshold 0.0 --number_of_matches 50'
```
If you run this you should get a Success rate of 60.1% and an SPL of 17.6%. This is different from the main paper as originally for NRNS we used Habitat version 0.1.6 which gives worse results.
To get results on straight replace ```--path_type=curved``` with ```--path_type=straight```. Furthermore, to get results on MP3D replace ```--dataset=gibson``` with ```--dataset=mp3d```.

### SLING + Straight x Gibson x Curved
```
python run.py --dataset=gibson --path_type=curved --difficulty=easy --distance_model_path=NRNS_ours.pt --model TopoGCNNRNS --tag straight_gibson_curved --use_glue_rhophi --dont_reuse_poses --switch_threshold 0.0 --number_of_matches 50 --straight_right_only 
```
If you run this you should get a Success rate of 39.2% and an SPL of 14.3%.
To get results on straight replace ```--path_type=curved``` with ```--path_type=straight```. Furthermore, to get results on MP3D replace ```--dataset=gibson``` with ```--dataset=mp3d```.


### NRNS x Gibson x Curved
```
python run.py --dataset=gibson --path_type=curved --difficulty=easy --distance_model_path=NRNS_ours.pt --model TopoGCNNRNS --tag nrns_gibson_curved --dont_reuse_poses'
```
If you run this you should get a Success rate of 27.5% and an SPL of 8.8%. In our paper, we use the numbers reported from [NRNSs' repository](https://github.com/meera1hahn/NRNS/)
To get results on straight replace ```--path_type=curved``` with ```--path_type=straight```. Furthermore, to get results on MP3D replace ```--dataset=gibson``` with ```--dataset=mp3d```.


## 📝 Citation
If you use this work, please cite:

```text
@inproceedings{
  wasserman2022lastmile,
  title={Last-Mile Embodied Visual Navigation},
  author={Justin Wasserman and Karmesh Yadav and Girish Chowdhary and Abhinav Gupta and Unnat Jain},
  booktitle={6th Annual Conference on Robot Learning},
  year={2022},
}
```
