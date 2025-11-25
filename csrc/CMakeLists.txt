cmake_minimum_required(VERSION 3.10)

project(PTAExtensionOPS)

execute_process(
  COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE python_site_packages_path
)
string(STRIP "${python_site_packages_path}" python_site_packages_path)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion-null")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fpie -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common")

set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wl,-Bsymbolic -rdynamic -Wl,--no-undefined ${CMAKE_CXX_FLAGS}")
set(PYTORCH_INSTALL_PATH ${python_site_packages_path}/torch)
set(PYTORCH_NPU_INSTALL_PATH ${python_site_packages_path}/torch_npu)

set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} \
    -pie -fPIE")

link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_INSTALL_PATH}/lib)
link_directories($ENV{ASCEND_HOME_PATH}/lib64)

add_library(PTAExtensionOPS SHARED
    ./plugin/register_ops.cpp
    ./plugin/rope.cpp
    ./plugin/la.cpp
    ./plugin/matmulv2.cpp
    ./plugin/batchmatmulv3.cpp
    ./plugin/batchmatmulv2.cpp
    ./plugin/batchmatmulv3duo.cpp
    ./plugin/la_preprocess.cpp)

target_compile_features(PTAExtensionOPS PRIVATE cxx_std_17)
if(DEFINED ENV{USER_ABI_VERSION})
    string(STRIP "$ENV{USER_ABI_VERSION}" ABI)
    if(NOT (ABI STREQUAL "0" OR ABI STREQUAL "1"))
        message(FATAL_ERROR "The value of USER_ABI_VERSION must be 0 or 1, but current value is '${ABI}'")
    endif()
else()
    set(ABI 0)
endif()
target_compile_options(PTAExtensionOPS PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${ABI})

include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/hccl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)

target_link_libraries(PTAExtensionOPS PUBLIC c10 torch torch_cpu torch_npu ascendcl)
