{ pkgs }:

let
  cuda = pkgs.cudaPackages_12_8.cudatoolkit;
  gcc = pkgs.gcc13;
  ffmpeg = pkgs.ffmpeg_7;
  ffmpegLib = ffmpeg.lib;
  caBundle = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
in
{
  packages = [
    pkgs.cmake
    pkgs.ninja
    pkgs.pkg-config
    gcc
    pkgs.cacert
    pkgs.linuxHeaders
    ffmpeg
    cuda
  ];

  env = {
    CUDA_HOME = "${cuda}";
    CUDA_PATH = "${cuda}";
    CUDACXX = "${cuda}/bin/nvcc";
    CUDAHOSTCXX = "${gcc}/bin/g++";
    CC = "${gcc}/bin/gcc";
    CXX = "${gcc}/bin/g++";
    CMAKE_CUDA_HOST_COMPILER = "${gcc}/bin/gcc";
    NVCC_PREPEND_FLAGS = "-ccbin ${gcc}/bin/gcc";
    SSL_CERT_FILE = caBundle;
    NIX_SSL_CERT_FILE = caBundle;
    REQUESTS_CA_BUNDLE = caBundle;
    CURL_CA_BUNDLE = caBundle;
    GIT_SSL_CAINFO = caBundle;
  };

  prependEnv = {
    PATH = "${ffmpeg}/bin";
    CPATH = "${pkgs.linuxHeaders}/include:${cuda}/include";
    LIBRARY_PATH = "${cuda}/lib64:${cuda}/lib";
    LD_LIBRARY_PATH = "${ffmpegLib}/lib:${cuda}/lib64:${cuda}/lib:${pkgs.stdenv.cc.cc.lib}/lib";
  };

  shellHook = ''
    export CFLAGS="-I${pkgs.linuxHeaders}/include ''${CFLAGS:-}"
    export CXXFLAGS="-I${pkgs.linuxHeaders}/include ''${CXXFLAGS:-}"
  '';
}
