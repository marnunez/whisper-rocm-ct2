# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.3
FROM ${BASE_IMAGE} AS build

ARG CT2_REPO=https://github.com/OpenNMT/CTranslate2.git
ARG CT2_REF=v4.7.1
ARG ROCM_ARCHS=gfx1100
ARG CMAKE_BUILD_PARALLEL_LEVEL=8
ENV DEBIAN_FRONTEND=noninteractive
ENV CTRANSLATE2_ROOT=/opt/ctranslate2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    hipblas-dev \
    hipcub-dev \
    hiprand-dev \
    libopenblas-dev \
    ninja-build \
    rocprim-dev \
    rocrand-dev \
    rocthrust-dev \
    patch \
    python3-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
      build \
      numpy \
      pybind11 \
      pyyaml \
      setuptools \
      wheel

# Build latest upstream CTranslate2 from source in the same ROCm family we run
# in to avoid libhipblas/libamdhip ABI mismatches. No PyTorch base is needed.
RUN git clone --recursive ${CT2_REPO} /tmp/CTranslate2 \
    && cd /tmp/CTranslate2 \
    && git checkout ${CT2_REF} \
    && git submodule update --init --recursive \
    && cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
      -DROCM_PATH=/opt/rocm \
      -DWITH_CUDA=OFF \
      -DWITH_CUDNN=OFF \
      -DWITH_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCHS} \
      -DGPU_TARGETS=${ROCM_ARCHS} \
      -DWITH_MKL=OFF \
      -DWITH_DNNL=OFF \
      -DWITH_OPENBLAS=ON \
      -DOPENMP_RUNTIME=COMP \
      -DBUILD_TESTS=OFF \
      -DBUILD_CLI=OFF \
      -DENABLE_CPU_DISPATCH=OFF \
    && cmake --build build --parallel ${CMAKE_BUILD_PARALLEL_LEVEL} \
    && cmake --install build \
    && cd python \
    && CT2_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
       CPLUS_INCLUDE_PATH=${CTRANSLATE2_ROOT}/include \
       LIBRARY_PATH=${CTRANSLATE2_ROOT}/lib \
       python3 -m pip wheel --no-build-isolation --no-deps -w /tmp/wheels .

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CTRANSLATE2_ROOT=/opt/ctranslate2
ENV LD_LIBRARY_PATH=/opt/ctranslate2/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib:${LD_LIBRARY_PATH}
ENV ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
ENV CT2_CUDA_ALLOCATOR=cub_caching
ENV CT2_CUDA_CACHING_ALLOCATOR_CONFIG=4,3,12,419430400
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV WHISPER_MODEL=large-v3-turbo
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16
ENV HOST=0.0.0.0
ENV PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    hipblas \
    hiprand \
    libgomp1 \
    rocrand \
    libopenblas0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/ctranslate2 /opt/ctranslate2
COPY --from=build /tmp/wheels/*.whl /tmp/
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir /tmp/ctranslate2-*.whl \
    && python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/*.whl /tmp/requirements.txt

WORKDIR /app
COPY app /app/app
EXPOSE 8080
CMD ["python3", "-m", "app"]
