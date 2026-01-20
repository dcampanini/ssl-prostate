import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

def evaluate(
    y_det: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    y_true: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    sample_weight: "Optional[Iterable[float]]" = None,
    subject_list: Optional[Iterable[Hashable]] = None,
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = 'IoU',
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = 'max',
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
    num_parallel_calls: int = 3,
    verbose: int = 0,
) -> Metrics:
    """
    Evaluate 3D detection performance.

    Parameters:
    - y_det: iterable of all detection_map volumes to evaluate. Each detection map should a 3D volume
        containing connected components (in 3D) of the same confidence. Each detection map may contain
        an arbitrary number of connected components, with different or equal confidences.
        Alternatively, y_det may contain filenames ending in .nii.gz/.mha/.mhd/.npy/.npz, which will
        be loaded on-the-fly.
    - y_true: iterable of all ground truth labels. Each label should be a 3D volume of the same shape
        as the corresponding detection map. Alternatively, `y_true` may contain filenames ending in
        .nii.gz/.mha/.mhd/.npy/.npz, which should contain binary labels and will be loaded on-the-fly.
        Use `1` to encode ground truth lesion, and `0` to encode background.
    - sample_weight: case-level sample weight. These weights will also be applied to the lesion-level
        evaluation, with same weight for all lesion candidates of the same case.
    - subject_list: list of sample identifiers, to give recognizable names to the evaluation results.
    - min_overlap: defines the minimal required Intersection over Union (IoU) or Dice similarity
        coefficient (DSC) between a lesion candidate and ground truth lesion, to be counted as a true
        positive detection.
    - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
        May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
        provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.
    - case_confidence_func: function to derive case-level confidence from detection map. Default: max.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).
    - num_parallel_calls: number of threads to use for evaluation. Set to 1 to disable parallelization.
    - verbose: (optional) controll amount of printed information.

    Returns:
    - Metrics
    """
    if sample_weight is None:
        sample_weight = itertools.repeat(1)
    if subject_list is None:
        # generate indices to keep track of each case during multiprocessing
        subject_list = itertools.count()

    # initialize placeholders
    case_target: Dict[Hashable, int] = {}
    case_weight: Dict[Hashable, float] = {}
    case_pred: Dict[Hashable, float] = {}
    lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
    lesion_weight: Dict[Hashable, List[float]] = {}

    # construct case evaluation kwargs
    evaluate_case_kwargs = dict(
        min_overlap=min_overlap,
        overlap_func=overlap_func,
        case_confidence_func=case_confidence_func,
        allow_unmatched_candidates_with_minimal_overlap=allow_unmatched_candidates_with_minimal_overlap,
        y_det_postprocess_func=y_det_postprocess_func,
        y_true_postprocess_func=y_true_postprocess_func,
    )

    with ThreadPoolExecutor(max_workers=num_parallel_calls) as pool:
        if num_parallel_calls >= 2:
            # process the cases in parallel
            futures = {
                pool.submit(
                    evaluate_case,
                    y_det=y_det_case,
                    y_true=y_true_case,
                    weight=weight,
                    idx=idx,
                    **evaluate_case_kwargs
                ): idx
                for (y_det_case, y_true_case, weight, idx) in zip(y_det, y_true, sample_weight, subject_list)
            }

            iterator = concurrent.futures.as_completed(futures)
        else:
            # process the cases sequentially
            def func(y_det_case, y_true_case, weight, idx):
                return evaluate_case(
                    y_det=y_det_case,
                    y_true=y_true_case,
                    weight=weight,
                    idx=idx,
                    **evaluate_case_kwargs
                )

            iterator = map(func, y_det, y_true, sample_weight, subject_list)

        if verbose:
            total: Optional[int] = None
            if isinstance(subject_list, Sized):
                total = len(subject_list)
            iterator = tqdm(iterator, desc='Evaluating', total=total)

        for result in iterator:
            if isinstance(result, tuple):
                # single-threaded evaluation
                lesion_results_case, case_confidence, weight, idx = result
            elif isinstance(result, concurrent.futures.Future):
                # multi-threaded evaluation
                lesion_results_case, case_confidence, weight, idx = result.result()
            else:
                raise TypeError(f'Unexpected result type: {type(result)}')

            # aggregate results
            case_weight[idx] = weight
            case_pred[idx] = case_confidence
            if len(lesion_results_case):
                case_target[idx] = np.max([a[0] for a in lesion_results_case])
            else:
                case_target[idx] = 0

            # accumulate outputs
            lesion_results[idx] = lesion_results_case
            lesion_weight[idx] = [weight] * len(lesion_results_case)

    # collect results in a Metrics object
    metrics = Metrics(
        lesion_results=lesion_results,
        case_target=case_target,
        case_pred=case_pred,
        case_weight=case_weight,
        lesion_weight=lesion_weight,
    )

    return metrics