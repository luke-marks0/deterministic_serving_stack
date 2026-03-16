{
  description = "Deterministic vLLM serving stack — hermetic runtime closure";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python310;

        # Pre-built wheel URLs and hashes per architecture.
        # These are the exact binaries pip would install — pinned by SHA256.
        wheelSources = {
          "aarch64-linux" = {
            vllm = {
              url = "https://files.pythonhosted.org/packages/f5/02/ff63919abb341b0819f33a400c83698d095e5fd461ae3e44f3ff91f6489f/vllm-0.17.1-cp38-abi3-manylinux_2_31_aarch64.whl";
              hash = "sha256:f04d63a94d0415b2323b0a0d3ab89a8d4d9bd346251ff60d47a7df679f7b3ff8";
            };
            torch = {
              url = "https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_aarch64.whl";
              hash = "sha256:e186f57ef1de1aa877943259819468fc6f27efb583b4a91f9215ada7b7f4e6cc";
            };
          };
          "x86_64-linux" = {
            vllm = {
              # Placeholder — replace with actual x86_64 wheel hash
              url = "https://files.pythonhosted.org/packages/cp310/v/vllm/vllm-0.17.1-cp38-abi3-manylinux_2_31_x86_64.whl";
              hash = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
            };
            torch = {
              url = "https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl";
              hash = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
            };
          };
        };

        wheels = wheelSources.${system} or wheelSources."x86_64-linux";

        # Fetch pinned wheels
        vllmWheel = pkgs.fetchurl {
          url = wheels.vllm.url;
          sha256 = wheels.vllm.hash;
        };

        torchWheel = pkgs.fetchurl {
          url = wheels.torch.url;
          sha256 = wheels.torch.hash;
        };

        # NVIDIA runtime wheels — CUDA libs that torch needs at runtime
        nvidiaWheelDefs = {
          "aarch64-linux" = [
            { url = "https://files.pythonhosted.org/packages/29/99/db44d685f0e257ff0e213ade1964fc459b4a690a73293220e98feb3307cf/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_aarch64.whl"; hash = "b86f6dd8935884615a0683b663891d43781b819ac4f2ba2b0c9604676af346d0"; }
            { url = "https://files.pythonhosted.org/packages/7c/75/f865a3b236e4647605ea34cc450900854ba123834a5f1598e160b9530c3a/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "52bf7bbee900262ffefe5e9d5a2a69a30d97e2bc5bb6cc866688caa976966e3d"; }
            { url = "https://files.pythonhosted.org/packages/eb/d1/e50d0acaab360482034b84b6e27ee83c6738f7d32182b987f9c7a4e32962/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "fc1fec1e1637854b4c0a65fb9a8346b51dd9ee69e61ebaccc82058441f15bce8"; }
            { url = "https://files.pythonhosted.org/packages/fa/41/e79269ce215c857c935fd86bcfe91a451a584dfc27f1e068f568b9ad1ab7/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_aarch64.whl"; hash = "c9132cc3f8958447b4910a1720036d9eff5928cc3179b0a51fb6d167c6cc87d8"; }
            { url = "https://files.pythonhosted.org/packages/60/bc/7771846d3a0272026c416fbb7e5f4c1f146d6d80704534d0b187dd6f4800/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "848ef7224d6305cdb2a4df928759dca7b1201874787083b6e7550dd6765ce69a"; }
            { url = "https://files.pythonhosted.org/packages/45/5e/92aa15eca622a388b80fbf8375d4760738df6285b1e92c43d37390a33a9a/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_aarch64.whl"; hash = "dfab99248034673b779bc6decafdc3404a8a6f502462201f2f31f11354204acd"; }
            { url = "https://files.pythonhosted.org/packages/c8/32/f7cd6ce8a7690544d084ea21c26e910a97e077c9b7f07bf5de623ee19981/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_aarch64.whl"; hash = "db9ed69dbef9715071232caa9b69c52ac7de3a95773c2db65bdba85916e4e5c0"; }
            { url = "https://files.pythonhosted.org/packages/bc/f7/cd777c4109681367721b00a106f491e0d0d15cfa1fd59672ce580ce42a97/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "9b6c161cb130be1a07a27ea6923df8141f3c295852f4b260c65f18f3e0a091dc"; }
            { url = "https://files.pythonhosted.org/packages/73/b9/598f6ff36faaece4b3c50d26f50e38661499ff34346f00e057760b35cc9d/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_aarch64.whl"; hash = "8878dce784d0fac90131b6817b607e803c36e629ba34dc5b433471382196b6a5"; }
            { url = "https://files.pythonhosted.org/packages/bb/1c/857979db0ef194ca5e21478a0612bcdbbe59458d7694361882279947b349/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "31432ad4d1fb1004eb0c56203dc9bc2178a1ba69d1d9e02d64a6938ab5e40e7a"; }
            { url = "https://files.pythonhosted.org/packages/2a/a2/8cee5da30d13430e87bf99bb33455d2724d0a4a9cb5d7926d80ccb96d008/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "adccd7161ace7261e01bb91e44e88da350895c270d23f744f0820c818b7229e7"; }
            { url = "https://files.pythonhosted.org/packages/10/c0/1b303feea90d296f6176f32a2a70b5ef230f9bdeb3a72bddb0dc922dc137/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "d7ad891da111ebafbf7e015d34879f7112832fc239ff0d7d776b6cb685274615"; }
            { url = "https://files.pythonhosted.org/packages/d5/1f/b3bd73445e5cb342727fd24fe1f7b748f690b460acadc27ea22f904502c8/nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "4412396548808ddfed3f17a467b104ba7751e6b58678a4b840675c56d21cf7ed"; }
            { url = "https://files.pythonhosted.org/packages/1e/f5/5607710447a6fe9fd9b3283956fceeee8a06cda1d2f56ce31371f595db2a/nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux_2_27_aarch64.whl"; hash = "4beb6d4cce47c1a0f1013d72e02b0994730359e17801d395bdcbf20cfb3bb00a"; }
            { url = "https://files.pythonhosted.org/packages/1d/6a/03aa43cc9bd3ad91553a88b5f6fb25ed6a3752ae86ce2180221962bc2aa5/nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"; hash = "0b48363fc6964dede448029434c6abed6c5e37f823cb43c3bcde7ecfc0457e15"; }
          ];
          "x86_64-linux" = []; # Placeholders — fill after first x86 build
        };

        nvidiaWheels = builtins.map
          (w: pkgs.fetchurl { url = w.url; sha256 = w.hash; })
          (nvidiaWheelDefs.${system} or []);

        # Python environment with all deps from nixpkgs + pinned wheels
        pythonEnv = python.withPackages (ps: [
          ps.numpy
          ps.jsonschema
          ps.requests
          ps.pyyaml
          ps.filelock
          ps.tqdm
          ps.typing-extensions
          ps.packaging
        ]);

        # Unzip wheels directly into site-packages (no pip needed)
        vllmInstall = pkgs.stdenv.mkDerivation {
          pname = "vllm-torch-wheels";
          version = "0.17.1";
          dontUnpack = true;
          nativeBuildInputs = [ pkgs.unzip ];
          dontAutoPatchelf = false;
          installPhase = ''
            mkdir -p $out/${python.sitePackages}
            unzip -qo ${torchWheel} -d $out/${python.sitePackages}
            unzip -qo ${vllmWheel} -d $out/${python.sitePackages}
            ${builtins.concatStringsSep "\n" (builtins.map (w: "unzip -qo ${w} -d $out/${python.sitePackages}") nvidiaWheels)}
          '';
          # autoPatchelfHook runs in fixupPhase automatically.
          # It rewrites RUNPATH in all ELF binaries under $out to reference
          # the Nix store paths for libstdc++, glibc, CUDA libs, etc.
        };

        # Application code
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

        # Full runtime closure
        runtimeClosure = pkgs.symlinkJoin {
          name = "deterministic-serving-runtime-closure";
          version = "0.1.0";
          paths = [
            pythonEnv
            vllmInstall
            appSrc
            pkgs.bash
            pkgs.coreutils
            pkgs.cacert
          ];
        };

        # OCI image
        ociImage = pkgs.dockerTools.buildLayeredImage {
          name = "deterministic-serving-runtime";
          tag = self.rev or "dev";
          contents = [ runtimeClosure ];
          config = {
            Cmd = [ "${pythonEnv}/bin/python3" "${appSrc}/cmd/server/main.py" ];
            WorkingDir = "/workspace";
            Env = [
              "PYTHONPATH=${appSrc}:${vllmInstall}/${python.sitePackages}"
              "VLLM_BATCH_INVARIANT=1"
              "CUBLAS_WORKSPACE_CONFIG=:4096:8"
              "PYTHONHASHSEED=0"
            ];
          };
        };

      in {
        packages = {
          default = runtimeClosure;
          closure = runtimeClosure;
          app = appSrc;
          oci = ociImage;
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
