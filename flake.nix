{
  description = "SVD-project nix flake";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = {nixpkgs, ...}: let
    systems = [
      "x86_64-darwin"
      "x86_64-linux"
      "aarch64-darwin"
      "aarch64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
    forAllPkgs = f: forAllSystems (system: let pkgs = nixpkgs.legacyPackages.${system}; in f pkgs);
  in {
    devShells = forAllPkgs (pkgs: {
      default = let
        libs =
          [
            pkgs.lapack-reference
            pkgs.eigen
            pkgs.boost
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            pkgs.llvmPackages.libcxx
          ];
      in
        pkgs.mkShell.override {stdenv = pkgs.llvmPackages.stdenv;} {
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.gfortran
            pkgs.llvmPackages.clang
            pkgs.ninja
            pkgs.pkg-config
            pkgs.clang-tools

            (pkgs.writeShellScriptBin "build" ''cmake -B build -G "Ninja" && cmake --build build'')
          ];
          buildInputs = libs;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        };
    });
  };
}
