{ lib, pkgs }:

let
  exportEnv = env:
    lib.concatStringsSep "\n" (
      lib.mapAttrsToList (name: value: "export ${name}=${lib.escapeShellArg value}") env
    );

  prependEnv = env:
    lib.concatStringsSep "\n" (
      lib.mapAttrsToList (
        name: value:
          ''
            _runtime_current_value="$(printenv ${name} || true)"
            if [ -n "$_runtime_current_value" ]; then
              export ${name}=${lib.escapeShellArg value}:"$_runtime_current_value"
            else
              export ${name}=${lib.escapeShellArg value}
            fi
            unset _runtime_current_value
          ''
      ) env
    );

  mergeStringAttrs = runtimes: key:
    lib.foldl' (
      acc: runtime:
        let current = runtime.${key} or { };
        in acc // lib.mapAttrs (
          name: value:
            if builtins.hasAttr name acc
            then "${value}:${builtins.getAttr name acc}"
            else value
        ) current
    ) { } runtimes;

  mergeRuntimes = runtimes: {
    packages = lib.unique (lib.concatMap (runtime: runtime.packages or [ ]) runtimes);
    env = lib.foldl' (acc: runtime: acc // (runtime.env or { })) { } runtimes;
    prependEnv = mergeStringAttrs runtimes "prependEnv";
    shellHook = lib.concatStringsSep "\n\n" (
      lib.filter (chunk: chunk != "") (map (runtime: runtime.shellHook or "") runtimes)
    );
  };

  mkRuntimeShell = {
    name,
    runtimes,
    extraPackages ? [ ],
    extraShellHook ? "",
    stdenv ? null,
  }:
    let
      runtime = mergeRuntimes runtimes;
      shellHook = lib.concatStringsSep "\n\n" (
        [
          (exportEnv (runtime.env or { }))
          (prependEnv (runtime.prependEnv or { }))
          (runtime.shellHook or "")
          extraShellHook
          ''
            echo "${name} shell ready"
          ''
        ]
      );
    in
    pkgs.mkShell ({
      packages = lib.unique (extraPackages ++ runtime.packages);
      inherit shellHook;
    } // lib.optionalAttrs (stdenv != null) { inherit stdenv; });
in
{
  inherit exportEnv prependEnv mergeRuntimes mkRuntimeShell;
}
