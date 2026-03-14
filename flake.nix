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
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python310;

        # Pinned Python packages via pip2nix-style overrides.
        # These SHAs are locked to exact versions for reproducibility.
        pythonEnv = python.withPackages (ps: with ps; [
          # Core serving stack
          ps.torch       # PyTorch (CUDA build via nixpkgs cudaSupport)
          ps.numpy
          ps.jsonschema
          ps.requests
          ps.huggingface-hub
          ps.safetensors
          ps.transformers
          ps.tokenizers
          ps.pyyaml
          ps.filelock
          ps.tqdm
          ps.typing-extensions
          ps.packaging
        ]);

        # vLLM is not in nixpkgs — build from pinned source.
        vllm = python.pkgs.buildPythonPackage rec {
          pname = "vllm";
          version = "0.17.1";
          format = "wheel";

          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/cp310/v/vllm/vllm-${version}-cp310-cp310-manylinux1_x86_64.whl";
            # Placeholder — replace with actual hash after first build
            hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
          };

          propagatedBuildInputs = with python.pkgs; [
            torch numpy transformers tokenizers safetensors
            huggingface-hub requests pyyaml tqdm packaging
          ];

          # Skip import check since it needs CUDA at import time
          pythonImportsCheck = [];
        };

        # The hermetic runtime closure: everything needed to run the server.
        runtimeClosure = pkgs.symlinkJoin {
          name = "deterministic-serving-runtime-closure";
          version = "0.1.0";
          paths = [
            pythonEnv
            pkgs.bash
            pkgs.coreutils
          ];
        };

        # OCI image built from the closure, pinned by digest.
        ociImage = pkgs.dockerTools.buildLayeredImage {
          name = "deterministic-serving-runtime";
          tag = self.rev or "dev";
          contents = [
            runtimeClosure
            (pkgs.writeTextDir "app/cmd" "")
          ];

          extraCommands = ''
            mkdir -p app
            cp -r ${self}/cmd app/
            cp -r ${self}/pkg app/
            cp -r ${self}/schemas app/
            cp -r ${self}/manifests app/ 2>/dev/null || true
          '';

          config = {
            Cmd = [ "${pythonEnv}/bin/python3" "/app/cmd/server/main.py" ];
            WorkingDir = "/workspace";
            Env = [
              "PYTHONPATH=/app"
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
          oci = ociImage;
        };

        # `nix develop` shell for local development
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
