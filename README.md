# 项目简介

本仓库大部分代码来自《CUDA C编程：权威指南》一书，同时加入了一些自己封装的工具类。

本项目利用CMake构建，具体编译安装过程如下。

# 安装步骤

克隆本仓库到本地

```sh
git clone https://github.com/Deleter-D/CUDA.git
```

如果你配置了SSH，则可以用以下命令代替

```sh
git clone git@github.com:Deleter-D/CUDA.git
```

___

使用CMake构建

```sh
cd build
cmake ..
```
> 请确保你的CMake版本大于等于3.17

---

构建完成后，使用make命令编译并安装

```sh
make
make install
```

> 你可以在make过程中使用`make -j32`命令开启多线程

---

安装完成后就可以在根目录的`executable`目录下看到所有可执行文件

```sh
cd ../executable
```

下面是运行其中一个可执行文件的例子

```sh
./01_programming_model/01_hello_world
```