{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python312Packages.numpy
  ];
  shellHook = ''
    # exported variables maybe...
  '';
}
