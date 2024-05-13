import onnxruntime as ort
import torch
import torch.onnx
import os
from main import import_class
model_name="VSL_SAM_SLR_V2_joint"
Model=import_class("model.decouple_gcn_attn.Model")
model=Model(num_class=100, num_point=27, num_person=1, graph="graph.sign_27.Graph", groups=16,block_size=41,graph_args={'labeling_mode': 'spatial'},in_channels=3).to('cuda')
model.load_state_dict(torch.load("save_models/vsl_joint_best_model.pt"))
dummy_input = torch.randn(1,3,120,27,1).to('cuda')
output_path = f'{model_name}.onnx'
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=['x'],
    output_names=['logits'],
    dynamic_axes={
        'x': {
            0: 'batch_size',
            1: 'num_channels',
            2: 'window_size',
            3: 'num_joints',
            4: 'max_num_bodies',
        },
    
    },
    opset_version=12
)