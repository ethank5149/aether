# Project AETHER development container.
#
# Shared by both checkouts: the public kernel builds it, and the controlled
# layer references the built image by tag rather than duplicating this file.
#
# Two conda environments on purpose, mirroring the host layout this replaces:
#
#   `aether`  the Python environment the project runs in
#   `su2`     the compiled solver and *its own* Open MPI
#
# SU2 cannot be pip-installed and its Open MPI must be the one it was linked
# against -- launching an Open MPI binary under a foreign `mpirun` fails in
# ways that read as a solver problem. The code already accounts for this:
# `find_mpirun` looks for `mpirun` *beside* the SU2 binary before it looks on
# PATH. Merging the two environments would break that invariant, so they stay
# apart and only the solver's bin directory is appended to PATH.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# System libraries only. The build toolchain is here for compiling anything
# staged under /workspace/source (SU2, Mutation++); the X11 and GL libraries
# are what gmsh's headless meshing links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        ca-certificates \
        pkg-config \
        less \
        vim-tiny \
        libglu1-mesa \
        libgl1 \
        libxft2 \
        libxinerama1 \
        libxcursor1 \
        libxrender1 \
        libxfixes3 \
    && rm -rf /var/lib/apt/lists/*

# Match the Unraid share owner (abc:users = 99:100). Without this the container
# runs as root and every file it writes into the bind-mounted shares comes back
# root-owned on the host, which is tedious to undo and easy to not notice until
# something outside the container cannot read its own results. GID 100 already
# exists in Ubuntu as `users`; UID 99 is free.
#
# Created *before* conda, and conda is then installed as this user, so the
# environments are user-owned from the start. The alternative -- installing as
# root and chowning afterwards -- rewrites every file in a multi-gigabyte tree
# and doubles the image, because a changed owner is a changed layer.
ARG USERNAME=aether
ARG USER_UID=99
ARG USER_GID=100
RUN if ! getent group ${USER_GID} >/dev/null; then groupadd -g ${USER_GID} ${USERNAME}; fi \
    && useradd -u ${USER_UID} -g ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && mkdir -p /opt/conda /workspace \
    && chown ${USER_UID}:${USER_GID} /opt/conda /workspace

USER ${USERNAME}

# Miniforge: conda-forge by default, no channel negotiation, no Anaconda ToS.
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -u -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml \
    && mamba clean --all -f -y

# The solver, in its own prefix with its own Open MPI.
RUN mamba create -n su2 -c conda-forge -y su2 \
    && mamba clean --all -f -y

# `SU2_RUN` is SU2's own convention for the directory its binaries live in, and
# is the contract the code resolves against -- not this image's paths, which
# nothing outside this file knows. `find_mpirun` then looks for the matching
# launcher beside that binary, which is why the solver's Open MPI does not need
# to be on PATH and must not shadow anything that is.
ENV SU2_RUN="/opt/conda/envs/su2/bin"

# `aether` first so its python wins; `su2` appended for interactive use only.
ENV PATH="/opt/conda/envs/aether/bin:${PATH}:/opt/conda/envs/su2/bin"
ENV CONDA_DEFAULT_ENV=aether

# The results and data shares, bind-mounted beside the workspace rather than
# inside it. Set here as well so a plain `docker run` of this image behaves the
# way the devcontainer does.
ENV AETHER_RESULTS="/results"

# The open reference datasets (GMTED2010 elevation, Blue Marble imagery, ERA5),
# bind-mounted at /data. `aether.viz.terrain` checks this before
# falling back to walking up from its own module -- a fallback that only worked
# while the code happened to sit inside a checkout that also held the data. With
# the data now on its own share, the variable is the mechanism, not a hint:
# unset, 24 viz tests fail on a missing elevation archive, because the fallback
# warning is promoted to an error by `filterwarnings = ["error"]`.
ENV AETHER_DATA_ROOT="/data"

# Open MPI refuses to run as root without an explicit override, and oversubscribe
# keeps a sweep from aborting when `workers * ranks` briefly exceeds the core
# count. Set regardless of user so a `docker exec` as root behaves the same.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_rmaps_base_oversubscribe=yes \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_MCA_osc=^ucx

# conda-forge lays the CUDA headers out under `targets/<arch>/include`, not the
# `$PREFIX/include` that nvcc's own wrapper knows about and everything else
# assumes. nvcc therefore compiles fine with this unset while cupy -- which
# builds kernels at runtime with nvrtc -- fails at the first kernel launch with
# "Failed to find CUDA headers". Pointing CUDA_PATH at the real layout fixes
# cupy and is what any hand-built CUDA code in /workspace/source will expect.
ENV CUDA_PATH="/opt/conda/envs/aether/targets/x86_64-linux"
ENV CUDA_HOME="${CUDA_PATH}"

# Headless: matplotlib and gmsh must not look for a display.
ENV MPLBACKEND=Agg

WORKDIR /workspace
