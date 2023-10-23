# Project Introduction

The majority of the code in this repository is derived from the book *Professional CUDA C Programming*, with the addition of some custom utility classes.

This project is built using CMake. The specific compilation and installation process is outlined below.

# Installation Steps

Clone this repository to your local machine.

```sh
git clone https://github.com/Deleter-D/CUDA.git
```

If you have SSH configured, you can use the following command instead.

```sh
git clone git@github.com:Deleter-D/CUDA.git
```

---

Use CMake to build.

```sh
cd build
cmake ..
```

> Please ensure that your CMake version is greater than or equal to 3.17.

---

Once the build is complete, use the make command to compile and install.

```sh
make
make install
```

> You can use the make -j32 command during the make process to enable multithreading.

---

After installation is complete, you can find all executable files in the executable directory at the root.

```sh
cd ../executable
```
Below is an example of running one of the executable files.

```sh
./01_programming_model/01_hello_world
```