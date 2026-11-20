import hashlib
import os
from io import BytesIO
from pathlib import Path

import onnx
import onnxruntime as rt

from .device import ORTDeviceInfo


def _get_coreml_cache_dir() -> Path:
    path = Path.home() / 'Library' / 'Caches' / 'DeepFaceLive' / 'CoreML'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_coreml_model(onnx_model_or_path) -> bytes:
    """
    Prepare model for CoreML MLProgram:
     - symbolic batch dimension of inputs is fixed to 1,
       MLProgram does not support unbounded dims (huge compile time / CPU fallback)
     - COREML_CACHE_KEY is set, so compiled model is cached on disk between runs
    """
    if isinstance(onnx_model_or_path, (str, Path)):
        st = os.stat(onnx_model_or_path)
        key_src = f'{os.path.abspath(onnx_model_or_path)}|{st.st_size}|{st.st_mtime_ns}'.encode()
        model = onnx.load(str(onnx_model_or_path))
    else:
        key_src = onnx_model_or_path
        model = onnx.load_from_string(onnx_model_or_path)

    initializer_names = { x.name for x in model.graph.initializer }
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) != 0 and dims[0].dim_value == 0:
            dims[0].dim_value = 1

    for prop in list(model.metadata_props):
        if prop.key == 'COREML_CACHE_KEY':
            model.metadata_props.remove(prop)
    prop = model.metadata_props.add()
    prop.key = 'COREML_CACHE_KEY'
    prop.value = hashlib.sha1(key_src).hexdigest()

    return model.SerializeToString()


def InferenceSession_with_device(onnx_model_or_path, device_info : ORTDeviceInfo):
    """
    Construct onnxruntime.InferenceSession with this Device.

     device_info     ORTDeviceInfo

    can raise Exception
    """

    if isinstance(onnx_model_or_path, onnx.ModelProto):
        b = BytesIO()
        onnx.save(onnx_model_or_path, b)
        onnx_model_or_path = b.getvalue()

    device_ep = device_info.get_execution_provider()
    if device_ep not in rt.get_available_providers():
        raise Exception(f'{device_ep} is not avaiable in onnxruntime')

    ep_flags = {}
    if device_ep in ['CUDAExecutionProvider','DmlExecutionProvider']:
        ep_flags['device_id'] = device_info.get_index()

    sess_options = rt.SessionOptions()
    sess_options.log_severity_level = 4
    sess_options.log_verbosity_level = -1
    if device_ep == 'DmlExecutionProvider':
        sess_options.enable_mem_pattern = False
    if device_ep == 'CoreMLExecutionProvider':
        # Default NeuralNetwork format computes in float16 on GPU/ANE and produces garbage for some models (DFM).
        # MLProgram is accurate, but the first compilation is slow, so it is cached on disk.
        onnx_model_or_path = _prepare_coreml_model(onnx_model_or_path)
        ep_flags['ModelFormat'] = 'MLProgram'
        ep_flags['MLComputeUnits'] = 'ALL'
        ep_flags['ModelCacheDirectory'] = str(_get_coreml_cache_dir())

    providers = [ (device_ep, ep_flags) ]
    if device_ep == 'CoreMLExecutionProvider':
        # ops not supported by CoreML are executed by CPU
        providers.append('CPUExecutionProvider')
    try:
        sess = rt.InferenceSession(onnx_model_or_path, providers=providers, sess_options=sess_options)
    except Exception:
        if device_ep != 'CoreMLExecutionProvider':
            raise
        # some models cannot be converted to MLProgram, fallback to NeuralNetwork format
        ep_flags['ModelFormat'] = 'NeuralNetwork'
        sess = rt.InferenceSession(onnx_model_or_path, providers=providers, sess_options=sess_options)
    return sess
