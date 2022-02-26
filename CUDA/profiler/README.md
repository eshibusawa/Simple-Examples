# Usage
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES:STRING=61
```
- The sample code is taken from [Using Nsight Compute to Inspect your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)
- GP10x (CC 6.1) is supported by [Nsight Compute-2019.5](https://developer.nvidia.com/nsight-compute-2019_5)
- Volta: Xavier (CC 7.2) is supported by [Nsight Compute-2020.1](https://developer.nvidia.com/nsight-compute-2020_1)
- Nsight Compute does NOT support Jetson Nano (CC 5.3)

# System setting
Configure nvidia driver [to allow any user to access GPU performance counter](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) and configure sshd to permit root login.
```sh
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"'> /etc/modprobe.d/nsight-privilege.conf
vi /etc/ssh/sshd_config
```
```diff
-#PermitRootLogin prohibit-password
+PermitRootLogin without-password
```
Generate a new SSH key `id_rsa.pub` and transfer to `/root/.ssh/authorized_keys`.
```sh
ssh-keygen -t rsa
```
