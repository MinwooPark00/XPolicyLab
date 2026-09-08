{
  description = "EgoVLA development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/15c6719d8c604779cf59e03c245ea61d3d7ab69b";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
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
          mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"
        '';
      };
    };
}
