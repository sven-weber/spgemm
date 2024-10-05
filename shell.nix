with import <nixpkgs> { };

pkgs.mkShell {
  name = "netsec-project";
  buildInputs = with pkgs; [
    gcc14
    clang-tools
    bear
    gdb
    boost
    gnumake
    openmpi
    pkg-config
  ];
}
