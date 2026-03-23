{
  description = "Deterministic vLLM serving stack — fully hermetic, built from source";

  # ───────────────────────────────────────────────────────────────────────────
  # Build strategy
  # ───────────────────────────────────────────────────────────────────────────
  # This flake builds PyTorch and vLLM entirely from source inside Nix so that
  # every shared library (including CUDA kernels) links against Nix's glibc,
  # libstdc++, and CUDA toolkit.  No manylinux wheels, no autoPatchelfHook,
  # no FHS escape hatches.
  #
  # Trade-off: a clean build takes 30–60 min on a beefy machine (torch alone
  # is ~20 min with parallelism).  Subsequent builds hit the Nix store cache.
  #
  # Placeholder hashes are marked "TODO: replace after first build" — Nix will
  # tell you the correct hash on the first attempt.
  # ───────────────────────────────────────────────────────────────────────────

  inputs = {
    # nixos-unstable has better CUDA / torch support than 24.11, especially
    # on aarch64-linux where 24.11's torch lacks CUDA entirely.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        # ── nixpkgs with CUDA enabled ──────────────────────────────────────
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;          # CUDA is unfree
            cudaSupport = true;          # propagate to all packages
            cudaCapabilities = [ "9.0" ]; # H100 / Hopper — trim fat from other archs
          };
        };

        python = pkgs.python312;
        pythonPackages = pkgs.python312Packages;

        # ── PyTorch from source (via nixpkgs) ─────────────────────────────
        torch = pythonPackages.torch;

        # ── vLLM 0.17.1 from source ───────────────────────────────────────
        vllmSrc = pkgs.fetchFromGitHub {
          owner = "vllm-project";
          repo = "vllm";
          rev = "v0.17.1";
          hash = "sha256-EZozwA+GIjN8/CBNhtdeM3HsPhVdx1/J0B9gvvn2qKU=";
          fetchSubmodules = true;
        };

        # ── Pre-fetched C++ dependencies for vLLM's cmake FetchContent ────
        # The Nix sandbox blocks network access during builds, so we
        # pre-fetch every repo that CMakeLists.txt would git-clone via
        # FetchContent, then point the corresponding *_SRC_DIR env vars
        # at the Nix store paths.

        cutlassSrc = pkgs.fetchFromGitHub {
          owner = "nvidia";
          repo = "cutlass";
          rev = "v4.2.1";
          hash = "sha256-iP560D5Vwuj6wX1otJhwbvqe/X4mYVeKTpK533Wr5gY=";
        };

        vllmFlashAttnSrc = pkgs.fetchgit {
          url = "https://github.com/vllm-project/flash-attention.git";
          rev = "140c00c0241bb60cc6e44e7c1be9998d4b20d8d2";
          hash = "sha256-GgLNpj44O2p6iitmSW82bENdS0tOmfdccngNlr4cKVY=";
          fetchSubmodules = true;
        };

        tritonSrc = pkgs.fetchFromGitHub {
          owner = "triton-lang";
          repo = "triton";
          rev = "v3.6.0";
          # TODO: replace after first build
          hash = "sha256-JFSpQn+WsNnh7CAPlcpOcUp0nyKXNbJEANdXqmkt4Tc=";
        };

        flashmlaSrc = pkgs.fetchFromGitHub {
          owner = "vllm-project";
          repo = "FlashMLA";
          rev = "692917b1cda61b93ac9ee2d846ec54e75afe87b1";
          # TODO: replace after first build
          hash = "sha256-GH7X25dy/PQiLIsItEzNa/N5r8VmOQRilIWLJdHj7kE=";
          fetchSubmodules = true;
        };

        qutlassSrc = pkgs.fetchFromGitHub {
          owner = "IST-DASLab";
          repo = "qutlass";
          rev = "830d2c4537c7396e14a02a46fbddd18b5d107c65";
          # TODO: replace after first build
          hash = "sha256-wXCQ5XlV8rKmctYCKDBc2aUqmHZX8qwXgGZY2BGyw5I=";
          fetchSubmodules = true;
        };

        vllm = pythonPackages.buildPythonPackage rec {
          pname = "vllm";
          version = "0.17.1";
          format = "setuptools";

          # Prevent Nix cmake hook from running its own configure phase.
          # vLLM's setup.py invokes cmake itself with the correct flags.
          dontUseCmakeConfigure = true;

          src = vllmSrc;

          # ── Build-time dependencies ────────────────────────────────────
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.which
            pkgs.git            # setup.py shells out to git for version
            pythonPackages.setuptools
            pythonPackages.setuptools-scm
            pythonPackages.wheel
            pythonPackages.packaging
          ];

          # ── Propagated runtime + build dependencies ────────────────────
          buildInputs = [
            # CUDA toolkit components
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.cuda_nvcc
            pkgs.cudaPackages.cuda_nvrtc
            pkgs.cudaPackages.cuda_cupti
            pkgs.cudaPackages.libcublas
            pkgs.cudaPackages.libcusolver
            pkgs.cudaPackages.libcusparse
            pkgs.cudaPackages.libcufft
            pkgs.cudaPackages.libcurand
            pkgs.cudaPackages.nccl
            pkgs.cudaPackages.cudnn

            # System libs
            pkgs.stdenv.cc.cc.lib  # libstdc++
            pkgs.zlib
            pkgs.openssl
          ];

          propagatedBuildInputs = [
            torch
            pythonPackages.numpy
            pythonPackages.transformers
            pythonPackages.tokenizers
            pythonPackages.sentencepiece
            pythonPackages.huggingface-hub
            pythonPackages.safetensors
            pythonPackages.requests
            pythonPackages.pyyaml
            pythonPackages.tqdm
            pythonPackages.filelock
            pythonPackages.typing-extensions
            pythonPackages.packaging
            pythonPackages.psutil
            pythonPackages.py-cpuinfo
            pythonPackages.pydantic
            pythonPackages.fastapi
            pythonPackages.uvicorn
            pythonPackages.uvloop
            pythonPackages.prometheus-client
            pythonPackages.aiohttp
            pythonPackages.ray
            pythonPackages.msgpack
            pythonPackages.pillow
            pythonPackages.openai
            pythonPackages.cbor2
            pythonPackages.gguf
            pythonPackages.pyzmq
            pythonPackages.httptools
            pythonPackages.python-dotenv
            pythonPackages.compressed-tensors
            pythonPackages.blake3
            pythonPackages.partial-json-parser
            pythonPackages.scipy
            pythonPackages.jinja2
            pythonPackages.lm-format-enforcer or null
            pythonPackages.outlines or null
          ];

          # ── Build environment ──────────────────────────────────────────
          env = {
            CUDA_HOME = "${pkgs.cudaPackages.cuda_nvcc}";
            TORCH_CUDA_ARCH_LIST = "9.0";
            MAX_JOBS = "8";
            SETUPTOOLS_SCM_PRETEND_VERSION = version;
            VLLM_PYTHON_EXECUTABLE = "${python}/bin/python3";
            # Pre-fetched sources for cmake FetchContent (no network in sandbox)
            VLLM_CUTLASS_SRC_DIR = "${cutlassSrc}";
            VLLM_FLASH_ATTN_SRC_DIR = "${vllmFlashAttnSrc}";
            # triton_kernels expects the full triton repo; cmake uses SOURCE_SUBDIR
            # but with TRITON_KERNELS_SRC_DIR it points directly to the python/triton_kernels/triton_kernels subdir
            TRITON_KERNELS_SRC_DIR = "${tritonSrc}/python/triton_kernels/triton_kernels";
            FLASH_MLA_SRC_DIR = "${flashmlaSrc}";
            QUTLASS_SRC_DIR = "${qutlassSrc}";
          };

          # vLLM's setup.py invokes cmake directly; we need CUDA on PATH
          preBuild = ''
            export PATH="${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH"
            export CUDA_HOME="${pkgs.cudaPackages.cuda_nvcc}"
            export CMAKE_PREFIX_PATH="${torch}/${python.sitePackages}/torch/share/cmake:$CMAKE_PREFIX_PATH"
            export VLLM_PYTHON_EXECUTABLE="${python}/bin/python3"
            export VLLM_CUTLASS_SRC_DIR="${cutlassSrc}"
            export VLLM_FLASH_ATTN_SRC_DIR="${vllmFlashAttnSrc}"
            export TRITON_KERNELS_SRC_DIR="${tritonSrc}/python/triton_kernels/triton_kernels"
            export FLASH_MLA_SRC_DIR="${flashmlaSrc}"
            export QUTLASS_SRC_DIR="${qutlassSrc}"
          '';

          # Skip tests — they require a live GPU
          doCheck = false;

          postFixup = ''
            # vLLM installs some scripts; ensure they point to our python
            for f in $out/bin/*; do
              if [ -f "$f" ]; then
                substituteInPlace "$f" \
                  --replace "/usr/bin/env python" "${python}/bin/python3" || true
              fi
            done
          '';

          meta = with pkgs.lib; {
            description = "High-throughput LLM serving engine";
            homepage = "https://github.com/vllm-project/vllm";
            license = licenses.asl20;
          };
        };

        # ── Python environment ─────────────────────────────────────────────
        pythonEnv = python.withPackages (ps: [
          torch
          vllm
          ps.numpy
          ps.jsonschema
          ps.requests
          ps.pyyaml
          ps.huggingface-hub
          ps.filelock
          ps.tqdm
          ps.typing-extensions
          ps.packaging
        ]);

        # ── Application source ─────────────────────────────────────────────
        appSrc = pkgs.stdenv.mkDerivation {
          pname = "deterministic-serving-stack";
          version = "0.1.0";
          src = self;
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            cp -r cmd $out/cmd
            cp -r pkg $out/pkg
            cp -r schemas $out/schemas
            cp -r manifests $out/manifests 2>/dev/null || true
          '';
        };

        # ── Full runtime closure ───────────────────────────────────────────
        runtimeClosure = pkgs.symlinkJoin {
          name = "deterministic-serving-runtime-closure";
          version = "0.1.0";
          paths = [
            pythonEnv
            appSrc
            pkgs.bash
            pkgs.coreutils
            pkgs.cacert
          ];
        };

        # ── OCI image ──────────────────────────────────────────────────────
        ociImage = pkgs.dockerTools.buildLayeredImage {
          name = "deterministic-serving-runtime";
          tag = self.rev or "dev";
          contents = [ runtimeClosure (pkgs.writeTextDir "etc/passwd" "root:x:0:0:root:/tmp:/bin/bash\n") (pkgs.writeTextDir "etc/group" "root:x:0:\n") ];
          config = {
            Cmd = [ "${pythonEnv}/bin/python3" "${appSrc}/cmd/server/main.py" ];
            WorkingDir = "/workspace";
            Env = [
              "PYTHONPATH=${appSrc}:${pythonEnv}/${python.sitePackages}"
              "VLLM_BATCH_INVARIANT=1"
              "CUBLAS_WORKSPACE_CONFIG=:4096:8"
              "PYTHONHASHSEED=0"
              "NVIDIA_VISIBLE_DEVICES=all"
              "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
              "LD_LIBRARY_PATH=/usr/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu"
              "HOME=/tmp"
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            ];
          };
        };

      in {
        packages = {
          default = runtimeClosure;
          closure = runtimeClosure;
          app = appSrc;
          oci = ociImage;
          inherit torch vllm;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.bash
            pkgs.jq
            pkgs.ripgrep
          ];
          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';
        };
      }
    );
}
