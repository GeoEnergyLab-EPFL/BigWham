## API

```mermaid
---
title: BigWham API
---
classDiagram
    note "From Duck till Zebra"
    Animal <|-- Duck
    note for Duck "can fly\ncan swim\ncan dive\ncan help in debugging"
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class BoundaryElement{
        +getCollocationPoints()
    }
    class Segment{
        -int sizeInFeet
        -canEat()
    }
    class Triangle{
        +bool is_wild
        +run()
    }
```


## Prerequisites

- Install Intel MKL
-- https://github.com/eddelbuettel/mkl4deb.git
-- directly run the script given in the above link
   
- Integrating cmake and intel MKL
-- https://cmake.org/cmake/help/latest/module/FindBLAS.html#intel-mkl
-- keep it in your bashrc
`. /opt/intel/mkl/bin/mklvars.sh intel64`

## making python venv

```
$sudo apt install python3-venv 

$python3 –m venv bigwham_venv 

$source bigwham_venv/bin/activate 

$pip install jupyter scipy matplotlib numpy jupyterlab 
```

## Compile

- add pybind11 submodule
  `git submodule update --init`

- compile command assuming `${PROJECT_ROOR}/bigwham_venv` is python virtual env
  run the following from project_root
```
source bigwham_venv/bin/activate;. /opt/intel/mkl/bin/mklvars.sh intel64; cd build; cmake -DPYENV_ROOT=${PWD}/bigwham_venv -DMATHEMATICA_INTERFACE=1 -DCMAKE_CXX_COMPILER=g++ -DBUILD_GOOGLE_TESTS=1 -DBUILD_PYTHON_BINDINGS=0 -DUSE_INTEL=0 -DIL_OPENMP=1 -DIL_OPENBLAS=0 -DIL_MKL=1  ..; make -j2; cd ..
```

## Testing

-`./build/BigWhamElastUnitTest`
- `python interfaces/python/_test/pybinding_checks.py` # inside virthual env



## Mathematica interface

- Now mathematica interface is made in the build folder, for example `${PROJECT_ROOT}/build` is the root folder, then the paclet file will be at `${PROJECT_ROOT}/build/interfaces/mathematica`
- All the variables will be already set if you compile the BigWham library correctly
- Switch on compiler flag `-DMATHEMATICA_INTERFACE=1` while using cmake
- We use `configure_file` in cmake to change the variables in `BigWhamSetting.m` file
- cmake ensures that mathematica is always connected to the updated `bigwham.a` file

### OLD WAY

## find lib paths in your system
- use the following python script to find the file path
 https://github.com/albertz/system-tools/blob/master/bin/find-lib-in-path.py

- libraries in the `ld_library_path` can be identified using 
`/sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH)`



- `libmkl_intel_lp64.a`

`/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.a`

- `libmkl_core.a`

`/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/libmkl_core.a`

- `libmkl_sequential.a`

`/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/libmkl_sequential.a`


- `-ltbb, -ldl, -lpthread, -lm`

- `-m64`

- `/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64_lin/gcc4.7/libtbb.so.2`

### modify file `BuildSettings.m`
- modify `$MKLROOT` in line 9
- modify `$TBBROOT` in line 10
- modify `$TBBLIBPATH` in line 11 
- modify `$BigWhamDirectory` in line 22

### modify file `BigWhamLink.m`
- modify build folder in `$BigwhamLib` in line 87
- modify build folder in `libdir` in line 143
