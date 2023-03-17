import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    # Compute intersection
    intersection = [max(prediction_box[0],gt_box[0]),
                    max(prediction_box[1],gt_box[1]),
                    min(prediction_box[2],gt_box[2]),
                    min(prediction_box[3],gt_box[3])]

    # Compute union
    
    if intersection[0] > intersection[2] or intersection[1] > intersection[3]:
        return 0.0
    
    prediction_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
        
    iou = intersection_area / (prediction_area + gt_area - intersection_area + 1e-16) #1e-16 for numerical stability
    # information on how to calculate found here: https://www.youtube.com/watch?v=XXYG5ZWtjj0&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=2&ab_channel=AladdinPersson
    #END OF YOUR CODE
    

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    # YOUR CODE HERE
    if num_tp+num_fp == 0:
        return 1
    prec = num_tp / (num_tp + num_fp)
    return prec
    #END OF YOUR CODE

    #raise NotImplementedError


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    # YOUR CODE HERE
    if num_tp+num_fn == 0:
        return 0
    recall = num_tp / (num_tp + num_fn)
    return recall
    #END OF YOUR CODE

    #raise NotImplementedError


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # YOUR CODE HERE
    
    # Find all possible matches with a IoU >= iou threshold
    matched_boxes = {}
    for i in range(len(prediction_boxes)):
        for j in range(len(gt_boxes)):
            iou = calculate_iou(prediction_boxes[i],gt_boxes[j])
            

            if iou >= iou_threshold:
                
                if not i in matched_boxes:
                    matched_boxes[i] = []
                matched_boxes[i].append([iou, j])
           

    
    corresponding_gt = []
    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold
    for pred in matched_boxes.keys():
        matched_boxes[pred].sort(reverse=True,key=lambda x: x[0])
        corresponding_gt.append(gt_boxes[matched_boxes[pred][0][1]])
    
    corresponding_gt = np.array(corresponding_gt)
    #print(prediction_boxes, corresponding_truths)

    return prediction_boxes, corresponding_gt
    #END OF YOUR CODE



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    # YOUR CODE HERE
    matched_pred, matched_gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    FN = len(gt_boxes)-len(matched_gt)
    FP = len(prediction_boxes) - len(matched_gt)
    TP = len(matched_gt)
    
    
    

    #END OF YOUR CODE

    return {"true_pos": TP, "false_pos": FP, "false_neg": FN}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # YOUR CODE HERE
    total= {"tp" : 0, "fp" : 0, "fn" : 0}

    for pred_box, gt_box in zip(all_prediction_boxes, all_gt_boxes):
        individual_im_res = calculate_individual_image_result(pred_box, gt_box, iou_threshold)
        total["tp"] += individual_im_res["true_pos"]
        total["fp"] += individual_im_res["false_pos"]
        total["fn"] += individual_im_res["false_neg"]

    prec = calculate_precision(total["tp"], total["fp"], total["fn"])
    rec = calculate_recall(total["tp"], total["fp"], total["fn"])
    return prec, rec
    #END OF YOUR CODE


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = [] 
    recalls = []
    
    for tresh in confidence_thresholds:
        chosen_boxes_for_tresh = []
        for pred, score in zip(all_prediction_boxes, confidence_scores):
            chosen_boxes = []
            for i in range(len(pred)):
                if(score[i] >= tresh):
                    chosen_boxes.append(pred[i])
            chosen_boxes_for_tresh.append(np.array(chosen_boxes))
        
        prec, rec = calculate_precision_recall_all_images(chosen_boxes_for_tresh, all_gt_boxes, iou_threshold) 
        precisions.append(prec)
        recalls.append(rec)
        # END OF YOUR CODE

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    max_precisions = []

    for lvl in recall_levels:
        max_prec = 0
        for prec, rec in zip(precisions, recalls):
            if prec > max_prec and rec >= lvl:
                max_prec = prec
        max_precisions.append(max_prec)
        
    avg_prec = np.average(max_precisions)

    return avg_prec
    #END OF YOUR CODE


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))

def main():
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

if __name__ == "__main__":
    main()
