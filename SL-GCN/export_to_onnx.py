import os
import yaml
import time
import torch
import pickle
import argparse
from collections import OrderedDict
from main import init_seed, import_class

WINDOW_SIZE = 120
NUM_JOINTS = 27
NUM_CHANNELS = 3
MAX_BODY_TRUE = 1


def get_args() -> argparse.Namespace:
    '''
    Get the arguments from the command line.

    Returns
    -------
    argparse.Namespace
        The arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config file.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='onnx',
        help='Path to the config file.',
    )
    return parser.parse_args()


def print_log(message: str, print_time: bool = True) -> None:
    '''
    Print the message with the current time.

    Parameters
    ----------
    message : str
        The message to print.
    print_time : bool, default=True
        Whether to print the current time.
    '''
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        message = "[ " + localtime + " ] " + message
    print(message)


def load_model(config, device: str = 'cpu') -> torch.nn.Module:
    '''
    Load the model from the config.

    Parameters
    ----------
    config : dict
        The configuration file.
    device : str, default='cpu'
        The device to load the model.

    Returns
    -------
    torch.nn.Module
        The loaded model.
    '''
    Model = import_class(config['model'])
    model = Model(**config['model_args']).to(device)

    if config['weights']:
        print_log("Load weights from {}.".format(config['weights']))
        if ".pkl" in config['weights']:
            with open(config['weights'], "r") as f:
                weights = pickle.load(f)
        else:
            ckpt = torch.load(config['weights'])
            if "weights" in ckpt.keys():
                weights = torch.load(config['weights'])["weights"]
            else:
                weights = ckpt

        weights = OrderedDict(
            [
                [k.split("module.")[-1], v.to(device)]
                for k, v in weights.items()
            ]
        )

        for w in config.get("remove_weights", []):
            if weights.pop(w, None) is not None:
                print_log("Sucessfully Remove Weights: {}.".format(w))
            else:
                print_log("Can Not Remove Weights: {}.".format(w))

        try:
            model.load_state_dict(weights)
        except Exception:
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            print("Can not find these weights:")
            for d in diff:
                print("  " + d)
            state.update(weights)
            model.load_state_dict(state)

    return model


def export_to_onnx(
    model,
    input_shape: tuple,
    output_path: str,
    device: str = 'cpu',
) -> None:
    '''
    Export the model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        The model to export.
    input_shape : tuple
        The shape of the input tensor.
    output_path : str
        The path to save the ONNX file.
    device : str, default='cpu'
        The device to run the model.
    '''
    dummny_input = torch.randn(*input_shape).to(device)
    torch.onnx.export(
        model,
        dummny_input,
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
        do_constant_folding=True,
        opset_version=13,
    )


def main(args: argparse.Namespace) -> None:
    '''
    Main function to export the model to ONNX format.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
    '''
    device = 'cpu'
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(config=config, device=device)
    model.eval()

    export_to_onnx(
        model=model,
        input_shape=(1, NUM_CHANNELS, WINDOW_SIZE, NUM_JOINTS, MAX_BODY_TRUE),
        output_path=os.path.join(args.output_dir, f'{config["Experiment_name"]}.onnx'),
        device=device,
    )


if __name__ == "__main__":
    init_seed(0)
    args = get_args()
    main(args=args)