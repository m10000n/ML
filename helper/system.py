import platform
import subprocess

import cpuinfo
import GPUtil
import psutil


def get_cpu_info():
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "hw.memsize"], capture_output=True, text=True, check=True
            )

        except Exception as e:
            return f"Error fetching total RAM: {e}"

        total_bytes = int(result.stdout.split(":")[1].strip())
        total_ram = round(total_bytes / (1024**3), 2)  #
    else:
        memory_info = psutil.virtual_memory()
        total_ram = round(memory_info.total / (1024**3), 2)

    cpu_info = cpuinfo.get_cpu_info()
    return {
        "brand": cpu_info.get("brand_raw", "Unknown"),
        "architecture": cpu_info.get("arch", "Unknown"),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram": f"{total_ram} GB",
    }


def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
    except ValueError:
        # NVIDIA-SMI is available, but there are no GPUs
        return None

    gpus_ = []
    for gpu in gpus:
        gpus_.append(
            {"id": gpu.id, "name": gpu.name, "memory": f"{gpu.memoryTotal} MB"}
        )

    return gpus_


def get_num_physical_cores():
    return psutil.cpu_count(logical=False)


def get_num_logical_cores():
    return psutil.cpu_count(logical=True)
