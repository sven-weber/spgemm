with import <nixpkgs> { };

pkgs.mkShell {
  name = "netsec-project";
  buildInputs = with pkgs; [
    boost
    gnumake
  ];
}
