{
  description = "Flow — Local AI Workflow System";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;

        pythonEnv = python.withPackages (ps: with ps; [
          httpx
          rich
          prompt_toolkit
          # Dev tools
          pytest
          ruff
        ]);

        # The flow package itself
        flowPkg = python.pkgs.buildPythonPackage {
          pname = "flow";
          version = "0.1.0";
          src = ./.;
          format = "pyproject";
          nativeBuildInputs = [ python.pkgs.hatchling ];
          propagatedBuildInputs = with python.pkgs; [
            httpx
            rich
            prompt_toolkit
          ];
        };

      in {
        packages = {
          default = flowPkg;
          flow = flowPkg;
        };

        apps.default = {
          type = "app";
          program = "${flowPkg}/bin/flow";
        };

        devShells.default = pkgs.mkShell {
          name = "flow-dev";
          buildInputs = [
            pythonEnv
            pkgs.ollama
            pkgs.ruff
            pkgs.shellcheck
            pkgs.shfmt
            pkgs.lua-language-server
            pkgs.stylua
            pkgs.rust-analyzer
            pkgs.cargo
            pkgs.go
            pkgs.gopls
            pkgs.nodejs
            pkgs.clang-tools  # clangd + clang-format
            pkgs.git
          ];

          shellHook = ''
            echo "Flow dev environment loaded"
            echo "Python: $(python --version)"
            echo "Run: pip install -e . && flow"
          '';
        };
      }
    );
}
