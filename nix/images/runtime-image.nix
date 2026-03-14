# OCI image derivation.
#
# This is the legacy interface — prefer `nix build .#oci` via flake.nix.
{ pkgs ? import <nixpkgs> {} }:

let
  runtimeClosure = import ../packages/runtime-closure.nix { inherit pkgs; };
in
pkgs.dockerTools.buildLayeredImage {
  name = "deterministic-serving-runtime";
  tag = "0.1.0";
  contents = [ runtimeClosure pkgs.python310 pkgs.bash ];
  config = {
    Cmd = [ "${runtimeClosure}/bin/python3" "${runtimeClosure}/cmd/server/main.py" ];
    WorkingDir = "/workspace";
    Env = [
      "PYTHONPATH=/app"
      "VLLM_BATCH_INVARIANT=1"
      "CUBLAS_WORKSPACE_CONFIG=:4096:8"
      "PYTHONHASHSEED=0"
    ];
  };
}
