from typing import List, Union

import torch


class TorchDeviceInfo:
    """
    Represents picklable torch device info
    """
    def __init__(self, index=None, name=None, total_memory=None, backend='cuda'):
        self._index : int = index
        self._backend : str = backend
        self._name : str = name
        self._total_memory : int = total_memory

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)

    def is_cpu(self) -> bool: return self._index == -1

    def get_backend(self) -> str:
        return getattr(self, '_backend', 'cuda')

    def get_index(self) -> int:
        return self._index

    def get_name(self) -> str:
        if self.is_cpu():
            return 'CPU'
        return self._name

    def get_total_memory(self) -> int:
        if self.is_cpu():
            return 0
        return self._total_memory

    def __eq__(self, other):
        if self is not None and other is not None and isinstance(self, TorchDeviceInfo) and isinstance(other, TorchDeviceInfo):
            return self._index == other._index
        return False

    def __hash__(self):
        return self._index

    def __str__(self):
        if self.is_cpu():
            return "CPU"
        elif self.get_backend() == 'mps':
            return f"[{self._index}] {self._name} [MPS]"
        else:
            return f"[{self._index}] {self._name} [{(self._total_memory / 1024**3) :.3}Gb]"

    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()

_torch_devices = None

def get_cpu_device_info() -> TorchDeviceInfo: return TorchDeviceInfo(index=-1)

def get_device_info_by_index( index : int ) -> Union[TorchDeviceInfo, None]:
    """
        index       if -1, returns CPU Device info
    """
    if index == -1:
        return get_cpu_device_info()
    for device in get_available_devices_info(include_cpu=False):
        if device.get_index() == index:
            return device
    return None

def get_device(device_info : TorchDeviceInfo) -> torch.device:
    """
    get physical torch.device from TorchDeviceInfo
    """
    if device_info.is_cpu():
        return torch.device('cpu')
    if device_info.get_backend() == 'mps':
        return torch.device('mps')
    return torch.device(f'cuda:{device_info.get_index()}')
        
def get_available_devices_info(include_cpu=True, cpu_only=False) -> List[TorchDeviceInfo]:
    """
    returns a list of available TorchDeviceInfo
    """
    devices = []
    if not cpu_only:
        global _torch_devices
        if _torch_devices is None:
            
            _torch_devices = []
            for i in range (torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                _torch_devices.append ( TorchDeviceInfo(index=i, name=device_props.name, total_memory=device_props.total_memory))
            if len(_torch_devices) == 0 and torch.backends.mps.is_available():
                # macOS Apple Silicon GPU
                _torch_devices.append ( TorchDeviceInfo(index=0, name='Apple GPU', total_memory=0, backend='mps'))
        devices += _torch_devices

    if include_cpu:
        devices.append ( get_cpu_device_info() )

    return devices

