with import <nixpkgs> { };

stdenvNoCC.mkDerivation {
  name = "dphpc-project";
  buildInputs = with pkgs; [
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
  ];

  MAKEFLAGS = "-j16";
}
