{
  description = "Basic rust dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix.url = "github:nix-community/fenix";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, fenix, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ fenix.overlays.default ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rust-toolchain = pkgs.fenix.latest.withComponents [
          "cargo"
          "clippy"
          "rustc"
          "rustfmt"
          "rust-analyzer"
        ];
      in
      with pkgs;
      {
        devShells.default = mkShell {
          buildInputs = [
            cargo-outdated
            cargo-fuzz
            rust-toolchain
          ];
        };
      }
    );
}
