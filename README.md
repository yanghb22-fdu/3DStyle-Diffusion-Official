# 3DStyle-Diffusion-Official
Official codes and datasets for ACM MM23 paper "3DStyle-Diffusion: Pursuing Fine-grained Text-driven 3D Stylization with 2D Diffusion Models"

- [x] Training code and configs.
- [x] Objaverse-3DStyle Datasets. The Objaverse-3DStyle dataset is available at [here](https://drive.google.com/file/d/1wWTLRaCf1VEeFFZiaIGeAvapGCdHwkFu/view?usp=sharing).

### System Requirements

- Python >=3.7
- CUDA 11
- Nvidia GPU with 12 GB ram at least
- Open3d >=0.14.1
- the package of clip (https://github.com/openai/CLIP)

### Train
Call the below shell scripts to generate example styles. 
```bash
# candle 
./shells/candle-golden.sh
# silver ring
./shells/ring.sh
# gold ring
./shells/ring3.sh
# a red rose sitting in a white vase
./shells/rose-in-vase.sh
# red rose with green leaves
./shells/rose.sh
```
The outputs will be saved to `results/`

## Citation
```
@inproceedings{HaiboYangACMMM2023,
  title={3DStyle-Diffusion: Pursuing Fine-grained Text-driven 3D
Stylization with 2D Diffusion Models},
  author={Haibo Yang and Yang Chen and Yingwei Pan and Ting Yao and Zhineng Chen and Tao Mei},
  booktitle={ACM MM},
  year={2023}
}
```
