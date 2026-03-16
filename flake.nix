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
          installPhase = ''
            mkdir -p $out/${python.sitePackages}
            unzip -qo ${torchWheel} -d $out/${python.sitePackages}
            unzip -qo ${vllmWheel} -d $out/${python.sitePackages}
          '';
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
