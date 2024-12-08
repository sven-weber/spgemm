with import <nixpkgs> { };

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

    texlive.combined.scheme-full
  ];

  MAKEFLAGS = "-j16";
}
