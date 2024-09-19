import GPUtil
import os
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.printout import printover
from pypaq.mpython.mptools import sys_res_nfo
import time
import torch
from typing import Optional, Union, List

from torchness.base import TorchnessException


"""
devices: DevicesTorchness - parameter type
    - represents some devices (GPU / CPU)
    - compatible with torch.device (class) type
 
    ### ************************************************************************************** torchness representations
    
    # cuda
    int                                 (int)                    single (system) CUDA ID
    -int                                (int)                    AVAILABLE CUDA[-int] 
    [] (empty list)                     (list)                   all AVAILABLE CUDA
    
    # cpu
    None                                (NoneType)               single CPU core
    float                               (float)                  (0.0;1.0> - factor of system CPU cores
    'all'                               (str)                    all CPU cores
    
    ### ***************************************************************************************** PyTorch representation
    'cpu'                               (str)                    PyTorch CPU
    'cuda'                              (str)                    PyTorch GPU
    'cuda:0'                            (str)                    PyTorch GPU
    torch.device                        (<class 'torch.device'>) PyTorch device type
    
    [int,-int,[],None,str,torch.device] (list)                   list with mix of ALL above, possible repetitions
       
"""
DevicesTorchness: Union[int, None, float, str, torch.device, List[Union[int,None,float,str,torch.device]]] = -1



def get_cuda_mem():
    """ returns cuda memory size (system first device) """
    devs = GPUtil.getGPUs()
    return devs[0].memoryTotal if devs else 0


def get_available_cuda_id(
        mem_free: int=      4000,
        load_max: float=    1.0,
) -> List[int]:
    """ returns list of available GPUs ids, ordered by free memory
    device is available if:
    - has at least mem_free (MB)
    - load <= load_max """
    cuda_devices = [(device.id, int(device.memoryFree), device.load) for device in GPUtil.getGPUs()]
    cuda_devices = [d for d in cuda_devices if d[1] >= mem_free and d[2] <= load_max]
    cuda_devices.sort(key=lambda x:x[1], reverse=True)
    return [d[0] for d in cuda_devices]


def report_cuda() -> str:
    """ prints report of system cuda devices """
    rp = 'System CUDA devices:'
    for device in GPUtil.getGPUs():
        rp += f'\n > id: {device.id}, name: {device.name}, MEM: {int(device.memoryUsed)}/{int(device.memoryTotal)} (U/T)'
    return rp


def _get_devices_torchness(
        devices: DevicesTorchness=  -1,
        mem_free: int=              4000,
        load_max: float=            1.0,
        logger=                     None,
        loglevel=                   20,
) -> List[Union[int,None]]:
    """ returns torchness representation of given devices
    max_load and max_mem are used only when requesting AVAILABLE CUDA (with -int or []) """

    if not logger:
        logger = get_pylogger(name='_get_devices_torchness', level=loglevel)

    # first convert to list
    if type(devices) is not list:
        devices = [devices]

    cpu_count = sys_res_nfo()['cpu_count']
    logger.debug(f'got {cpu_count} CPU devices in a system')

    # try to get available CUDA
    available_cuda_id = []
    try:
        available_cuda_id = get_available_cuda_id(mem_free=mem_free, load_max=load_max)
    except Exception as e:
        logger.warning(f'could not get CUDA devices, got EXCEPTION: {e}')

    if devices == []:
        return available_cuda_id

    devices_base = []
    for d in devices:

        known_device = False

        if type(d) is int:
            known_device = True
            if d >= 0:
                devices_base.append(d)
            else:
                if available_cuda_id:
                    devices_base.append(available_cuda_id[d])

        if d == []:
            known_device = True
            devices_base += available_cuda_id

        if d is None:
            known_device = True
            devices_base.append(d)

        if type(d) is float:
            known_device = True
            if d < 0.0: d = 0.0
            if d > 1.0: d = 1.0
            cpu_count_f = round(cpu_count * d)
            if cpu_count_f < 1: cpu_count_f = 1
            devices_base += [None]*cpu_count_f

        if type(d) is torch.device:
            d = str(d)

        if type(d) is str:
            if d == 'all':
                known_device = True
                devices_base += [None]*cpu_count
            if 'cpu' in d:
                known_device = True
                devices_base.append(None)
            if 'cuda' in d:
                known_device = True
                dn = 0
                if ':' in d:
                    dn = int(d.split(':')[-1])
                devices_base.append(dn)

        if not known_device:
            msg = f'unknown (not valid?) device given: {d}'
            logger.error(msg)
            raise TorchnessException(msg)

    return devices_base


def get_devices(
        devices: DevicesTorchness=  -1,
        mem_free: int=              4000,
        load_max: float=            1.0,
        eventually_cpu: bool=       False,
        torch_namespace: bool=      True,
        logger=                     None,
        loglevel=                   20,
) -> List[Union[int,None,str]]:
    """ resolves representation given with devices (DevicesTorchness) """

    if not logger:
        logger = get_pylogger(name='get_devices', level=loglevel)

    devices_base = _get_devices_torchness(
        devices=    devices,
        mem_free=   mem_free,
        load_max=   load_max,
        logger=     logger)

    d = [f'cuda:{dev}' if type(dev) is int else 'cpu' for dev in devices_base] if torch_namespace else devices_base
    if not d and eventually_cpu:
        d = ['cpu']
    return d


def mask_cuda(ids: Optional[List[int] or int]=None):
    """ masks GPUs from given list of ids or single one """
    if ids is None:
        ids = []
    if type(ids) is int:
        ids = [ids]
    mask = ''
    for i in ids:
        mask += f'{int(i)},'
    if len(mask) > 1:
        mask = mask[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = mask


def mask_cuda_devices(
        devices: DevicesTorchness=  -1,
        logger=                     None):
    """ wraps mask_cuda to hold DevicesTorchness """
    devices = get_devices(devices, torch_namespace=False, logger=logger)
    ids = [d for d in devices if type(d) is int]
    mask_cuda(ids)


def monitor(pause:float=0.1, print_n:int=10):
    """ monitors GPUs usage and memory in the loop
    :param float pause: amount of time loop is paused
    :param int print_n: prints report every N loops
    """

    devs = GPUtil.getGPUs()
    peaks_load = {d.id: 0.0 for d in devs}
    peaks_mem =  {d.id: 0.0 for d in devs}

    pnix = 0
    while True:
        devs = GPUtil.getGPUs()
        s = '*** GPUs monitor: '
        for d in devs:

            _id = d.id
            load = int(d.load*100)
            mem = int(d.memoryUsed)

            if peaks_load[_id] < load:
                peaks_load[_id] = load

            if peaks_mem[_id] < mem:
                peaks_mem[_id] = mem

            s += f'{_id}:{load}%/{int(mem)}MB '
        s += f' >>> peak: '
        for _id in peaks_load:
            s += f'{_id}:{peaks_load[_id]}%/{peaks_mem[_id]}MB '


        time.sleep(pause)

        pnix += 1
        if pnix % print_n == 0:
            pnix = 0
            printover(s[:-1])
