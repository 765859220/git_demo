
# Q-Diffusion: Quantizing Diffusion Models [[website](https://xiuyuli.com/qdiffusion/)] [[paper](http://arxiv.org/abs/2302.04304)]



## Getting Started

### Installation

Clone this repository, and then create and activate a suitable conda environment named `qdiff` by using the following command:

```bash
git clone https://github.com/Xiuyu-Li/q-diffusion.git
cd q-diffusion
conda env create -f environment.yml
conda activate qdiff
```


### Calibration
To conduct the calibration process, you must first generate the corresponding calibration datasets. We provide some example calibration datasets [here](https://drive.google.com/drive/folders/12TVeziKWNz_HmTAIxQLDZlHE33PKdpb1?usp=sharing). These datasets contain around 1000-2000 samples of intermediate outputs at each time step, which are much more than sufficient for calibration purposes. We will soon upload smaller subsets that meet the minimum requirements for calibration. In the meantime, you may consider generating the calibration datasets yourself by following the procedures described in the paper.

To reproduce the calibrated checkpoints, you can use the following commands:

```bash
# Stable Diffusion
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit <4 or 8> --quant_mode qdiff --quant_act --act_bit 8 --cali_st 25 --cali_batch_size 8 --cali_n 128 --no_grad_ckpt --split --running_stat --sm_abit 16 --cali_data_path <cali_data_path> --outdir <output_path>

# VLDM
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff  --quant_act --act_bit 8 --cali_st 25 --cali_batch_size 4 --cali_n 75 --no_grad_ckpt --running_stat --sm_abit 16 --cali_data_path <cali_data_path> --ddim_steps 25