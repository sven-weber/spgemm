with import <nixpkgs> { config.allowUnfree = true; };

stdenvNoCC.mkDerivation {
  name = "dphpc-project";
  buildInputs = with pkgs; [
    cmake
    gnat14
    clang-tools
    bear
    gdb
    boost
    gnumake
    openmpi
    pkg-config

    python3
    python3Packages.pandas
    python3Packages.requests
    python3Packages.scipy
    python3Packages.matplotlib
    python3Packages.jupyter-core
    python3Packages.notebook
    python3Packages.pip
    python3Packages.ipykernel

    vscode
    texlive.combined.scheme-full
  ];

  MAKEFLAGS = "-j16";
  NIXOS_OZONE_WL="1";
}
