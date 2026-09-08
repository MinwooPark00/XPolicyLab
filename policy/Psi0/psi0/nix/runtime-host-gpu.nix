{ pkgs }:

{
  shellHook = ''
    _runtime_detect_and_prepend() {
      _runtime_var_name="$1"
      shift
      _runtime_joined_paths=""
      for _runtime_path in "$@"; do
        if [ -e "$_runtime_path" ]; then
          if [ -n "$_runtime_joined_paths" ]; then
            _runtime_joined_paths="$_runtime_joined_paths:$_runtime_path"
          else
            _runtime_joined_paths="$_runtime_path"
          fi
        fi
      done

      if [ -n "$_runtime_joined_paths" ]; then
        eval "_runtime_current_value=\''${$_runtime_var_name}"
        if [ -n "$_runtime_current_value" ]; then
          export "$_runtime_var_name=$_runtime_joined_paths:$_runtime_current_value"
        else
          export "$_runtime_var_name=$_runtime_joined_paths"
        fi
        unset _runtime_current_value
      fi

      unset _runtime_joined_paths
      unset _runtime_var_name
      unset _runtime_path
    }

    if [ -z "''${VK_ICD_FILENAMES:-}" ]; then
      for _runtime_vk_icd in \
        /run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json \
        /usr/share/vulkan/icd.d/nvidia_icd.json \
        /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
        /etc/vulkan/icd.d/nvidia_icd.json
      do
        if [ -f "$_runtime_vk_icd" ]; then
          export VK_ICD_FILENAMES="$_runtime_vk_icd"
          break
        fi
      done
      unset _runtime_vk_icd
    fi

    if [ -z "''${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]; then
      for _runtime_egl_vendor in \
        /run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json \
        /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        /usr/share/glvnd/egl_vendor.d/50_nvidia.json
      do
        if [ -f "$_runtime_egl_vendor" ]; then
          export __EGL_VENDOR_LIBRARY_FILENAMES="$_runtime_egl_vendor"
          break
        fi
      done
      unset _runtime_egl_vendor
    fi

    _runtime_detect_and_prepend LD_LIBRARY_PATH \
      /run/opengl-driver/lib \
      /run/opengl-driver-32/lib \
      /usr/lib/x86_64-linux-gnu/nvidia/current \
      /usr/lib/x86_64-linux-gnu/nvidia \
      /usr/lib/nvidia \
      /usr/lib/nvidia-570 \
      /usr/lib/nvidia-575 \
      /usr/lib/nvidia-580 \
      /usr/lib/nvidia-590 \
      /usr/lib/wsl/lib

    _runtime_detect_and_prepend XDG_DATA_DIRS \
      /run/opengl-driver/share \
      /usr/share

    if [ -z "''${TRITON_LIBCUDA_PATH:-}" ]; then
      for _runtime_triton_cuda in \
        /run/opengl-driver/lib \
        /usr/lib/x86_64-linux-gnu \
        /usr/lib/x86_64-linux-gnu/nvidia/current \
        /usr/lib/x86_64-linux-gnu/nvidia \
        /usr/lib/nvidia-590 \
        /usr/lib/nvidia-580 \
        /usr/lib/nvidia-575 \
        /usr/lib/nvidia-570 \
        /usr/lib/wsl/lib
      do
        if [ -d "$_runtime_triton_cuda" ]; then
          export TRITON_LIBCUDA_PATH="$_runtime_triton_cuda"
          break
        fi
      done
      unset _runtime_triton_cuda
    fi

    unset -f _runtime_detect_and_prepend
  '';
}
