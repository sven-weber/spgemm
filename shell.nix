with import <nixpkgs> { };

pkgs.mkShell {
  name = "netsec-project";
  buildInputs = with pkgs; [
    gcc14
    clang-tools
    bear
    boost
    gnumake
    openmpi
    pkg-config
  ];
}
