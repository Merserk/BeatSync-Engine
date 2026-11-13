"""One discrete GPU for all BeatSync hardware backends.

Use Vulkan's physical-device type, rather than shared memory size or a name
heuristic, to exclude integrated GPUs. Prefer NVIDIA when present, then the
discrete card with the most local memory. Backend indices are not interchangeable:
Vulkan and CUDA devices are matched by hardware UUID (or Windows LUID).
"""

import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


class GPUSelectionError(RuntimeError):
    pass


def _vulkan_devices():
    loader = ctypes.WinDLL('vulkan-1.dll')

    class ApplicationInfo(ctypes.Structure):
        _fields_ = [('sType', ctypes.c_uint32), ('pNext', ctypes.c_void_p),
                    ('name', ctypes.c_char_p), ('version', ctypes.c_uint32),
                    ('engine', ctypes.c_char_p), ('engineVersion', ctypes.c_uint32),
                    ('apiVersion', ctypes.c_uint32)]

    class InstanceCreateInfo(ctypes.Structure):
        _fields_ = [('sType', ctypes.c_uint32), ('pNext', ctypes.c_void_p),
                    ('flags', ctypes.c_uint32), ('app', ctypes.POINTER(ApplicationInfo)),
                    ('layerCount', ctypes.c_uint32), ('layers', ctypes.c_void_p),
                    ('extensionCount', ctypes.c_uint32), ('extensions', ctypes.c_void_p)]

    class IDProperties(ctypes.Structure):
        _fields_ = [('sType', ctypes.c_uint32), ('pNext', ctypes.c_void_p),
                    ('uuid', ctypes.c_ubyte * 16), ('driverUUID', ctypes.c_ubyte * 16),
                    ('luid', ctypes.c_ubyte * 8), ('nodeMask', ctypes.c_uint32),
                    ('luidValid', ctypes.c_uint32)]

    class Properties2(ctypes.Structure):
        # An aligned, oversized buffer accommodates VkPhysicalDeviceProperties.
        _fields_ = [('sType', ctypes.c_uint32), ('pNext', ctypes.c_void_p),
                    ('properties', ctypes.c_uint64 * 512)]

    class MemoryType(ctypes.Structure):
        _fields_ = [('flags', ctypes.c_uint32), ('heap', ctypes.c_uint32)]

    class MemoryHeap(ctypes.Structure):
        _fields_ = [('size', ctypes.c_uint64), ('flags', ctypes.c_uint32)]

    class MemoryProperties(ctypes.Structure):
        _fields_ = [('typeCount', ctypes.c_uint32), ('types', MemoryType * 32),
                    ('heapCount', ctypes.c_uint32), ('heaps', MemoryHeap * 16)]

    loader.vkCreateInstance.argtypes = [ctypes.POINTER(InstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    loader.vkCreateInstance.restype = ctypes.c_int32
    loader.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    loader.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    loader.vkGetPhysicalDeviceProperties2.argtypes = [ctypes.c_void_p, ctypes.POINTER(Properties2)]
    loader.vkGetPhysicalDeviceProperties2.restype = None
    loader.vkGetPhysicalDeviceMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryProperties)]
    loader.vkGetPhysicalDeviceMemoryProperties.restype = None
    loader.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    loader.vkDestroyInstance.restype = None
    instance = ctypes.c_void_p()
    app = ApplicationInfo(sType=0, name=b'BeatSync', apiVersion=(1 << 22) | (1 << 12))
    create = InstanceCreateInfo(sType=1, app=ctypes.pointer(app))
    if loader.vkCreateInstance(ctypes.byref(create), None, ctypes.byref(instance)) != 0:
        raise GPUSelectionError('The Vulkan loader could not create a Vulkan 1.1 instance')
    try:
        count = ctypes.c_uint32()
        if loader.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != 0 or not count.value:
            raise GPUSelectionError('No Vulkan physical devices found')
        handles = (ctypes.c_void_p * count.value)()
        if loader.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles) != 0:
            raise GPUSelectionError('Vulkan device enumeration failed')
        devices = []
        for index, handle in enumerate(handles[:count.value]):
            identity = IDProperties(sType=1000071004)
            properties = Properties2(sType=1000059001, pNext=ctypes.addressof(identity))
            loader.vkGetPhysicalDeviceProperties2(handle, ctypes.byref(properties))
            data = bytes(properties.properties)
            version, _, vendor, device_id, device_type = struct.unpack_from('5I', data)
            memory = MemoryProperties()
            loader.vkGetPhysicalDeviceMemoryProperties(handle, ctypes.byref(memory))
            devices.append({
                'index': index, 'name': data[20:276].split(b'\0', 1)[0].decode('utf-8', errors='replace'),
                'discrete': device_type == 2, 'type': device_type,
                'vendor_id': vendor, 'device_id': device_id,
                'uuid': bytes(identity.uuid).hex(),
                'luid': bytes(identity.luid).hex() if identity.luidValid else '',
                'local_memory_bytes': sum(h.size for h in memory.heaps[:memory.heapCount] if h.flags & 1),
                'version': f'{(version >> 22) & 127}.{(version >> 12) & 1023}.{version & 4095}',
            })
        return devices
    finally:
        loader.vkDestroyInstance(instance, None)


def _hardware_identity(device):
    for key in ('uuid', 'luid'):
        identity = device.get(key, '').lower()
        if identity and int(identity, 16):
            return key + ':' + identity
    return ''


def select_discrete_gpu(devices):
    """Discover a dGPU from driver properties, independent of its name/slot.

    GPU indices come from this run's enumeration. Hardware identity breaks
    equal vendor/memory ties so reordering adapters keeps the same selection.
    A failed backend never makes an integrated GPU eligible.
    """
    candidates = [d for d in devices if d['type'] == 2]
    if not candidates:
        raise GPUSelectionError('No discrete Vulkan GPU found; GPU acceleration requires a dGPU')
    return dict(max(candidates, key=lambda d: (d.get('vendor_id') == 0x10de,
                                             d.get('local_memory_bytes', 0),
                                             _hardware_identity(d), -d['index'])))


@lru_cache(maxsize=1)
def _selected_gpu():
    try:
        return dict(select_discrete_gpu(_vulkan_devices()), available=True)
    except (OSError, RuntimeError, AttributeError) as exc:
        return {'available': False, 'name': 'No discrete GPU', 'error': str(exc)}


def get_discrete_gpu():
    return dict(_selected_gpu())


def require_discrete_gpu():
    device = get_discrete_gpu()
    if not device['available']:
        raise GPUSelectionError(device['error'])
    return device


def ffmpeg_gpu_environment():
    """CUDA enumeration and FFmpeg must see the same, unremapped ordinals."""
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('CUDA_DEVICE_ORDER', None)
    return env


def llama_gpu_environment():
    env = os.environ.copy()
    device = get_discrete_gpu()
    # llama.cpp remaps the single visible physical device to Vulkan0. An
    # empty list hides every GPU when no discrete card is available.
    env['GGML_VK_VISIBLE_DEVICES'] = str(device['index']) if device['available'] else ''
    # Environment aliases must not override the explicit model/projector args.
    for key in ('LLAMA_ARG_DEVICE', 'LLAMA_ARG_N_GPU_LAYERS', 'LLAMA_ARG_SPLIT_MODE',
                'LLAMA_ARG_MAIN_GPU', 'LLAMA_ARG_TENSOR_SPLIT', 'MTMD_BACKEND_DEVICE',
                'LLAMA_ARG_MMPROJ_OFFLOAD'):
        env.pop(key, None)
    return env


def _cuda_devices_native():
    """Query the installed NVIDIA driver; no CUDA toolkit/package is needed."""
    driver = ctypes.WinDLL('nvcuda.dll')
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetLuid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
    for name in ('cuInit', 'cuDeviceGetCount', 'cuDeviceGet', 'cuDeviceGetName', 'cuDeviceGetUuid', 'cuDeviceGetLuid'):
        getattr(driver, name).restype = ctypes.c_int

    def check(code):
        if code:
            raise GPUSelectionError(f'NVIDIA driver device query failed (CUDA error {code})')

    check(driver.cuInit(0))
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)))
    devices = []
    for index in range(count.value):
        handle = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(handle), index))
        name = ctypes.create_string_buffer(256)
        uuid = (ctypes.c_ubyte * 16)()
        luid = (ctypes.c_ubyte * 8)()
        mask = ctypes.c_uint()
        check(driver.cuDeviceGetName(name, len(name), handle))
        check(driver.cuDeviceGetUuid(uuid, handle))
        luid_valid = driver.cuDeviceGetLuid(luid, ctypes.byref(mask), handle) == 0
        devices.append({'index': index, 'name': name.value.decode('utf-8', errors='replace'),
                        'uuid': bytes(uuid).hex(), 'luid': bytes(luid).hex() if luid_valid else ''})
    return devices


def match_cuda_device(selected, devices):
    if selected.get('vendor_id') != 0x10de:
        raise GPUSelectionError(f"The selected dGPU ({selected['name']}) does not support NVIDIA CUDA/NVENC")
    for key in ('uuid', 'luid'):
        identity = selected.get(key)
        if identity and int(identity, 16):
            matches = [d for d in devices if d.get(key) == identity]
            if len(matches) == 1:
                return dict(matches[0])
    # Names only suffice for an unambiguous card when identity queries were
    # unavailable. Never guess an ordinal for two identically named cards.
    def normalized(name):
        return name.lower().removeprefix('nvidia ').strip()
    matches = [d for d in devices if normalized(d['name']) == normalized(selected['name'])]
    if len(matches) == 1 and not any(selected.get(k) and all(d.get(k) for d in devices) for k in ('uuid', 'luid')):
        return dict(matches[0])
    raise GPUSelectionError(f"Could not match selected dGPU {selected['name']} to a CUDA device")


@lru_cache(maxsize=1)
def _selected_cuda_gpu():
    try:
        selected = require_discrete_gpu()
        if selected.get('vendor_id') != 0x10de:
            raise GPUSelectionError(f"The selected dGPU ({selected['name']}) does not support NVIDIA CUDA/NVENC")
        # A fresh process prevents inherited CUDA visibility/order settings or
        # another library's initialized CUDA runtime from changing the mapping.
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--cuda-devices'],
                                env=ffmpeg_gpu_environment(), capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=15,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise GPUSelectionError(result.stderr.strip()[-1200:] or 'CUDA device enumeration failed')
        return dict(match_cuda_device(selected, json.loads(result.stdout)), available=True)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': str(exc)}


def get_cuda_gpu():
    return dict(_selected_cuda_gpu())


def require_cuda_gpu():
    device = get_cuda_gpu()
    if not device['available']:
        raise GPUSelectionError(device['error'])
    return device


def _dxgi_devices_native():
    """DXGI adapter ordinals for Direct3D decoding, matched by Windows LUID."""
    class LUID(ctypes.Structure):
        _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_int32)]
    class Description(ctypes.Structure):
        _fields_ = [('name', ctypes.c_wchar * 128), ('vendor_id', ctypes.c_uint32),
                    ('device_id', ctypes.c_uint32), ('subsystem', ctypes.c_uint32), ('revision', ctypes.c_uint32),
                    ('video_memory', ctypes.c_size_t), ('system_memory', ctypes.c_size_t),
                    ('shared_memory', ctypes.c_size_t), ('luid', LUID), ('flags', ctypes.c_uint32)]
    import uuid
    iid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    factory = ctypes.c_void_p()
    create = ctypes.WinDLL('dxgi.dll').CreateDXGIFactory1
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_int32
    if create(iid, ctypes.byref(factory)) < 0:
        raise GPUSelectionError('DXGI adapter enumeration failed')
    def method(pointer, index, result, *arguments):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[index])
    devices = []
    try:
        enum = method(factory, 12, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        for index in range(64):
            adapter = ctypes.c_void_p()
            code = enum(factory, index, ctypes.byref(adapter))
            if code < 0:
                break
            try:
                description = Description()
                code = method(adapter, 10, ctypes.c_int32, ctypes.POINTER(Description))(adapter, ctypes.byref(description))
                if code < 0:
                    raise GPUSelectionError('DXGI adapter description failed')
                devices.append({'index': index, 'name': description.name, 'vendor_id': description.vendor_id,
                                'device_id': description.device_id,
                                'luid': struct.pack('<Ii', description.luid.low, description.luid.high).hex()})
            finally:
                method(adapter, 2, ctypes.c_uint32)(adapter)
    finally:
        method(factory, 2, ctypes.c_uint32)(factory)
    return devices


def match_dxgi_device(selected, devices):
    luid = selected.get('luid')
    matches = [d for d in devices if luid and d.get('luid') == luid]
    if len(matches) == 1:
        return dict(matches[0])
    if not luid or not int(luid, 16):
        matches = [d for d in devices if (d.get('vendor_id'), d.get('device_id')) ==
                   (selected.get('vendor_id'), selected.get('device_id'))]
        if len(matches) == 1:
            return dict(matches[0])
    raise GPUSelectionError(f"Could not match selected dGPU {selected['name']} to a DXGI adapter")


@lru_cache(maxsize=1)
def get_dxgi_gpu():
    try:
        return dict(match_dxgi_device(require_discrete_gpu(), _dxgi_devices_native()), available=True)
    except (OSError, RuntimeError, AttributeError) as exc:
        return {'available': False, 'error': str(exc)}


if __name__ == '__main__' and sys.argv[1:] == ['--cuda-devices']:
    try:
        print(json.dumps(_cuda_devices_native()))
    except (OSError, RuntimeError, AttributeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
