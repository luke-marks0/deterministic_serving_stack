{ pkgs ? import <nixpkgs> {} }:

let
  runtimeClosure = import ../packages/runtime-closure.nix { inherit pkgs; };
in
pkgs.dockerTools.buildLayeredImage {
  name = "deterministic-serving-runtime";
  tag = "0.1.0";
  contents = [ runtimeClosure pkgs.python3 pkgs.bash ];
  config = {
    Cmd = [ "python3" "/bin/cmd/runner/main.py" ];
    WorkingDir = "/workspace";
  };
}
