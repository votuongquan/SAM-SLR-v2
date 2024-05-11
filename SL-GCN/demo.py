import os
import yaml
import time
import logging
import argparse
import pandas as pd
import onnxruntime as ort
from mediapipe.python.solutions import holistic
from utils import get_predictions, preprocess


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def get_args() -> argparse.Namespace:
    '''
    Get the arguments from the command line.

    Returns
    -------
    argparse.Namespace
        The arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='config/demo.yaml',
        help='Path to the config file.',
    )
    return parser.parse_args()


def check_arg(
    condition: bool,
    message: str,
    is_terminated: bool = True,
) -> None:
    '''
    Check the condition and log the message if the condition is False.

    Parameters
    ----------
    condition : bool
        The condition to check.
    message : str
        The message to log.
    is_terminated : bool, default=True
        Whether to terminate the program if the condition is False.
    '''
    if not condition:
        logging.error(message)
        if is_terminated:
            exit(1)


def inference(
    source: str,
    keypoints_detector: holistic.Holistic,
    ort_session: ort.InferenceSession,
    id2gloss: dict,
) -> str:
    '''
    Video-based inference for Vietnamese Sign Language recognition.

    Parameters
    ----------
    source: str
        The path to the video.
    keypoints_detector: mediapipe.solutions.holistic.Holistic
        The keypoints detector.
    ort_session: ort.InferenceSession
        ONNX Runtime session.
    id2gloss: dict
        Mapping from class index to class label.

    Returns
    -------
    str
        The inference message.
    '''
    start_time = time.time()
    inputs = preprocess(
        source=source,
        keypoints_detector=keypoints_detector,
    )
    end_time = time.time()
    data_time = end_time - start_time
    logging.info(f'Data processing time: {data_time:.2f} seconds.')

    start_time = time.time()
    predictions = get_predictions(
        inputs=inputs, ort_session=ort_session, id2gloss=id2gloss, k=3
    )
    end_time = time.time()
    model_time = end_time - start_time
    logging.info(f'Model inference time: {model_time:.2f} seconds.')

    output_message = 'Inference results:\n'
    if len(predictions) == 0:
        output_message += 'No sign language detected in the video. Please try again.'
    else:
        output_message += 'The top-3 predictions are:\n'
        for i, prediction in enumerate(predictions):
            output_message += f'\t{i+1}. {prediction["label"]} ({prediction["score"]:2f})\n'
        output_message += f'Total time: {data_time + model_time:.2f} seconds.'

    return output_message


def main(args: argparse.Namespace) -> None:
    '''
    Main function for the demo.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments.
    '''
    check_arg(os.path.isfile(args.config), 'Configuration file does not exist.')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    is_saved = config.get('output_dir', None) is not None
    if is_saved:
        os.makedirs(config['output_dir'], exist_ok=True)

    keypoints_detector = holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True,
    )

    check_arg(os.path.isfile(config.get('model', '')), 'Model ONNX file does not exist.')
    ort_session = ort.InferenceSession(config['model'])

    check_arg(os.path.isfile(config.get('id2gloss', '')), 'id2gloss file does not exist.')
    id2gloss = pd.read_csv(config['id2gloss'], names=['id', 'gloss']).to_dict()['gloss']

    check_arg(
        condition=all([
            config.get('source', None) is not None,
            os.path.isfile(config.get('source', ''))
        ]),
        message='Video file does not exist. Webcam mode is used.',
        is_terminated=False,
    )
    inference_message = inference(
        source=config.get('source', 0),
        keypoints_detector=keypoints_detector,
        ort_session=ort_session,
        id2gloss=id2gloss,
    )
    logging.info(inference_message)


if __name__ == '__main__':
    args = get_args()
    main(args=args)