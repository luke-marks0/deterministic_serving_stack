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

        # Base Python environment: deps that Nix can build hermetically.
        # vLLM + PyTorch + CUDA are external artifacts pinned in the lockfile,
        # not in the Nix closure — they require CUDA toolkits and GPU-specific
        # compilation that doesn't fit the Nix model cleanly.
        pythonEnv = python.withPackages (ps: [
          ps.jsonschema
          ps.requests
          ps.pyyaml
          ps.filelock
          ps.tqdm
          ps.typing-extensions
          ps.packaging
          ps.numpy
        ]);

        # Application code as a derivation
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

        # The hermetic runtime closure: Python + system deps + our code.
        # This is what runtime_closure_digest is computed from.
        runtimeClosure = pkgs.symlinkJoin {
          name = "deterministic-serving-runtime-closure";
          version = "0.1.0";
          paths = [
            pythonEnv
            appSrc
            pkgs.bash
            pkgs.coreutils
            pkgs.cacert  # for HTTPS
          ];
        };

        # OCI image: the closure + entrypoint config.
        # vLLM/PyTorch/CUDA are expected to be in the base image or
        # volume-mounted — they're tracked by digest in the lockfile.
        ociImage = pkgs.dockerTools.buildLayeredImage {
          name = "deterministic-serving-runtime";
          tag = self.rev or "dev";
          contents = [ runtimeClosure ];
          config = {
            Cmd = [ "${pythonEnv}/bin/python3" "${appSrc}/cmd/server/main.py" ];
            WorkingDir = "/workspace";
            Env = [
              "PYTHONPATH=${appSrc}"
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
