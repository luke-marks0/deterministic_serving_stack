# Runtime closure derivation.
#
# This is the legacy interface — prefer `nix build .#closure` via flake.nix.
# Kept for compatibility with the builder's --nix-store-path workflow.
{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python310;

  pythonEnv = python.withPackages (ps: with ps; [
    torch numpy jsonschema requests huggingface-hub
    safetensors transformers tokenizers pyyaml filelock
    tqdm typing-extensions packaging
  ]);
in
pkgs.symlinkJoin {
  name = "deterministic-serving-runtime-closure";
  version = "0.1.0";
  paths = [
    pythonEnv
    pkgs.bash
    pkgs.coreutils
  ];

  meta = {
    description = "Hermetic runtime closure for deterministic vLLM serving";
  };
}
