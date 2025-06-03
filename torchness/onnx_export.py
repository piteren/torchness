import onnxruntime as ort
import torch
from typing import Optional, Dict, List

from torchness.nnmodule import NNModule


def get_tested_model(
        model: Optional[torch.nn.Module],
        model_class: Optional[type(NNModule)],
        model_ckpt_fp: Optional[str],
        inputs: Dict[str,torch.Tensor],
):

    if not model and (not model_class or not model_ckpt_fp):
        raise Exception('model OR NNModule type with ckpt must be given!')

    if not model:
        model = model_class.build(ckpt_fp=model_ckpt_fp, device='cpu')
    print(model)

    out = model(**inputs)

    if type(out) is list:
        print(f'model output ({len(out)}):')
        for v in out:
            v_type = type(v)
            s = f'{v_type}'
            if v_type is torch.Tensor:
                s += f' {v.shape}'
            print(s)
    if type(out) is dict:
        print(f'model output ({len(out)}):')
        for k,v in out.items():
            v_type = type(v)
            s = f'{k}: {v_type}'
            if v_type is torch.Tensor:
                s += f' {v.shape}'
            print(s)

    return model


def export(
        model: Optional[torch.nn.Module],
        model_class: Optional[type(NNModule)],
        model_ckpt_fp: Optional[str],
        onnx_model_path: str,
        inputs: Dict[str,torch.Tensor],
        output_names: List[str]):

    model = get_tested_model(model=model, model_class=model_class, model_ckpt_fp=model_ckpt_fp, inputs=inputs)

    input_names = list(inputs.keys())

    torch.onnx.export(
        model=          model,
        args=           tuple(inputs.values()),
        f=              onnx_model_path,
        export_params=  True,
        input_names=    input_names,
        output_names=   output_names,
        dynamic_axes=   {k: {0: 'batch_size'} for k in input_names + output_names},
    )


def play(onnx_model_path:str, inputs:Dict[str,torch.Tensor]):

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(
        path_or_bytes=  onnx_model_path,
        sess_options=   sess_options,
        providers=      providers)

    print("\nONNX Model Inputs:")
    for inp in sess.get_inputs():
        print(f"{inp.name}, Shape: {inp.shape}")
    print("ONNX Model Outputs:")
    for out in sess.get_outputs():
        print(f"{out.name}, Shape: {out.shape}")

    in_feed = {k: v.numpy() for k,v in inputs.items()}
    print(f'\nFeeding ONNX {onnx_model_path} with inputs ({len(in_feed)}):')
    for k,v in in_feed.items():
        print(f'{k:20}: {v.shape}')
    out = sess.run(output_names=None, input_feed=in_feed)
    print(f'\nOutput ({len(out)}):')
    for e in out:
        print(e.shape)