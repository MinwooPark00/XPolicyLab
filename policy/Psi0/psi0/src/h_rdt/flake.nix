{
  description = "H_RDT dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/15c6719d8c604779cf59e03c245ea61d3d7ab69b";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              git
              gnumake
              gcc13
              ninja
              pkg-config
              python311
              uv
            ];

            shellHook = ''
              export TMPDIR="''${TMPDIR:-$PWD/.tmp}"
              export UV_CACHE_DIR="''${UV_CACHE_DIR:-$PWD/.cache/uv}"
              export PIP_CACHE_DIR="''${PIP_CACHE_DIR:-$PWD/.cache/pip}"
              export TRITON_CACHE_DIR="''${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
              mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR"
            '';
          };
        });
    };
}
