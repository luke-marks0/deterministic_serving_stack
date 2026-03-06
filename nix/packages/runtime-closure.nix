{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation {
  pname = "deterministic-serving-runtime";
  version = "0.1.0";

  src = ../..;
  dontBuild = true;

  installPhase = ''
    mkdir -p "$out/bin"
    cp -r "$src/cmd" "$out/bin/cmd"
    cp -r "$src/pkg" "$out/bin/pkg"
    cp -r "$src/schemas" "$out/bin/schemas"
  '';

  meta = {
    description = "Reference runtime closure for deterministic serving stack";
  };
}
