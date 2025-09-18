# TAT: Task-Adaptive Transformer for All-in-One Medical Image Restoration

This paper has been early accepted by **MICCAI 2025 (top 9%)**

The full code is scheduled for release in **December 2025**.

## Network Architecture

![](README.assets\fig_framework.png)

## Dataset

You can download the preprocessed datasets for MRI super-resolution, CT denoising, and PET synthesis from [Baidu Netdisk](https://pan.baidu.com/s/1oBBG_Stcn7cfO8U49S146w?pwd=3x13) or [Google Drive](https://drive.google.com/drive/folders/12Qdkdms14Kfv3P60clyCURtnoFpLhFnX?usp=sharing).

The original dataset for MRI super-resolution and CT denoising are as follows:

- MRI super-resolution: [IXI dataset](http://brain-development.org/ixi-dataset/)

- CT denoising: [AAPM dataset](https://www.aapm.org/grandchallenge/lowdosect/)

## Visualization

You can use [AMIDE](https://amide.sourceforge.net/) to visualize the ".nii" file. Note that the color map for MRI and CT images is "black/white linear," while the color map for PET images is "white/black linear." Additionally, you need to rescale the PET image according to the voxel size specified in the paper.

![](README.assets\fig_vis.png)

## Citation

If you find **TAT** useful in your research, please consider citing:

```bibtex
@inproceedings{yang2025tat,
  title={TAT: Task-Adaptive Transformer for All-in-One Medical Image Restoration},
  author={Yang, Zhiwen and Zhang, Jiaju and Yi, Yang and Liang, Jian and Wei, Bingzheng and Xu, Yan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2025},
  organization={Springer}
}
```

