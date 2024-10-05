with import <nixpkgs> { };

pkgs.mkShell {
  name = "netsec-project";
  buildInputs = with pkgs; [
    clang-tools
    bear
    boost
    gnumake
    openmpi
  ];
}
