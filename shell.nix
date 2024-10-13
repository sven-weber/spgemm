with import <nixpkgs> { };

stdenvNoCC.mkDerivation {
  name = "netsec-project";
  buildInputs = with pkgs; [
    gnat14
    clang-tools
    bear
    gdb
    boost
    gnumake
    openmpi
    pkg-config
    libossp_uuid
  ];

  MAKEFLAGS = "-j16";
}
