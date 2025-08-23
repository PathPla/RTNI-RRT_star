# RTNI-RRT*

## 📋 Project Overview

This project implements **RTNI-RRT*** (Real-Time Neural Informed RRT*), an efficient path-planning algorithm. Compared with traditional methods, it enables on-the-fly planning during execution on large-scale maps, significantly reducing wait time.

### 🎯 Key Features

- **🚀** **Real-time:** Plan while executing for rapid response
- **🧠 Neural guidance:** PointNet/PointNet2 enables intelligent sampling
- **📈 Scalability:** Supports very large maps (1000×1000+)
- **⚡ High efficiency:** Our method achieves an average 58\% reduction in terms of iteration count and runtime compared to the NRRT* baseline
- **📊 Compute-friendly:** Fully separates computation time from execution time
- **🔄 Checkpoint & resume:** Data generation supports interruption recovery
- **📁 Complete dataset:** 2,250 environments and 4,150 test samples

## 🚀 Quick Start

### 1. Setup

```bash
conda env create -f environment.yml
conda activate rtni-rrt

```

### 2. Data Collection

* Collect 2D random world data

  ```python
  python generate_random_world_env_2d.py
  python generate_random_world_env_2d_point_cloud.py
  ```
* Generate block and gap environment configurations

  ```python
  python generate_block_gap_env_2d.py
  ```

### 3. Model Training

```python
python train_pointnet_pointnet2.py --random_seed 100 --model pointnet2 --dim 2
python eval_pointnet_pointnet2.py --random_seed 100 --model pointnet2 --dim 2
python train_unet.py
python eval_unet.py
```

### 4. Generate test data (resumable)

```bash
python generate_rtni_rrt_star_datasets.py
```

### 5. Algorithm Comparison Experiments

```bash
python eval_rtni_rrt_star.py --algorithms rtni_rrt_star --environments scaled_maps
python plot.py
```

## References

[zhm-real/PathPlanning](zhm-real/PathPlanning)

[tedhuang96/nirrt_star](https://github.com/tedhuang96/nirrt_star?tab=readme-ov-file)

## Contact

Please feel free to open an issue or send an email to [xlilixiao@stu.suda.edu.cn](mailto:xlilixiao@stu.suda.edu.cn).
