# SLING and OVRL-GD/DDPPO-GD
This repository includes the code that can be used to reproduce the results of our paper [Last-Mile Embodied Visual Navigation](https://jbwasse2.github.io/portfolio/SLING/). 

This branch contains code to run a __OVRL-GD and DDPPO-GD__ exploration with  SLING. If you are interested in SLING + NRNS-GD/Straight-GD, please see the Experiment Variations section.

## 🗺 Table of Contents
<div class="toc">
<ul>
<li><a href="#-experiment-variations">🔬 Experiment Variations</a></li>
<li><a href="#-installation"> 💿 Installation</a></li>
<li><a href="#-running-ddppo-and-ovrl-with-sling"> 🏃 Running DDPPO and OVRL with SLING</a></li>
<li><a href="#-citation"> 📝 Citation</a></li>
</ul>
</li>
</ul>
</div>



## 🔬 Experiment Variations
To reproduce the results from NRNS-GD/Straight-GD with SLING, go to the nrns branch.
```
git checkout nrns
```

## 💿 Installation

To run this codebase please follow the given steps:

1. Create a new conda env:
```
conda create -n habitat python=3.7 cmake=3.14.0
conda activate habitat
```

2. Install [`Habitat-Sim`](https://github.com/facebookresearch/habitat-sim) version 0.2.1:
```
conda install habitat-sim=0.2.1 withbullet headless -c conda-forge -c aihabitat
```

3. We provide a slightly modified version of [`Habitat-Lab`](https://github.com/facebookresearch/habitat-lab), install it using:
```
cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all # install habitat and habitat_baselines
```

4. Add the `src` folder in the pythonpath. This folder contains a few files for the localnav policy used in SLING.
```
export PYTHONPATH=${PYTHONPATH}:/path/to/src/folder/
```

5. Now download the checkpoint files required to run the evaluation over OVRL and DDPPO:
```
gdown 1OfJ3fMo7II1lNUZs743zalBMSV71JeDC
unzip checkpoint.zip

gdown 1KJ0XHLlg9CxgPBG7OAM1c8L9ZG4AKzxv
unzip data.zip

gdown 1bR4bH7-OrDqo7TItwBcg4BS0Ls08Z1vl
unzip models.zip
```

6. Move these folders to the appropriate locations
```
mv checkpoint/ovrl_best_run/chkp/best_ovrl_ckpt.pth embodied_ssl/checkpoint/ovrl_best_run/chkp/
mv data embodied_ssl/
```

7. We will also need the Gibson and MP3D scene datasets. Instructions to download the datasets are available [here](https://github.com/facebookresearch/habitat-lab#matterport3d) and [here](https://github.com/facebookresearch/habitat-lab#matterport3d). Move the scene_datasets into the corresponding location.
```
mv gibson_train_val embodied_ssl/data/scene_datasets/
mv mp3d embodied_ssl/data/scene_datasets/
```


8. To run the evaluation, first change the following variable in file `sbatch_scripts/run_script.sh` on line 5 to the actual path on your machine:
```
REPO_PATH=/path/to/code/embodied_ssl
```

## 🏃 Running DDPPO and OVRL with SLING

Finally run the following command to start the evaluation of `SLING + OVRL-GD` and `SLING + DDPPO-GD` on `easy`, `medium` and `hard` subsets of `gibson-straight`, `gibson-curved`, `matterport-straight` and `matterport-curved`.
```
./sbatch_scripts/run_ovrl.sh
./sbatch_scripts/run_ddppo.sh
```
You should get the following results for the `gibson-curved` x `easy` datasplit
```
Average episode reward: 2.4231
Average episode distance_to_goal: 1.2594
Average episode success: 0.6830
Average episode spl: 0.4567
Average episode softspl: 0.4091
```

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
