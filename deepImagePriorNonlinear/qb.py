import argparse

import numpy as np
import torch
from PIL import Image
from skimage import filters
from skimage import morphology
from sklearn.metrics import confusion_matrix, f1_score

from featureExtractionModule import deepPriorCd
from utilities import saturateSomePercentileBandwise

# Defining Parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manualSeed', type=int, default=40, help='manual seed')
opt = parser.parse_args()
manualSeed = opt.manualSeed
print('Manual seed is ' + str(manualSeed))

outputLayerNumbers = [5]

nanVar = float('nan')

# setting manual seeds

torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)

preChangeDataPath = '../datasets/QB_original/t1.bmp'
postChangeDataPath = '../datasets/QB_original/t2.bmp'
referencePath = '../datasets/QB_original/gt.bmp'
resultPath = '../results/QB/QBDeepImagePriorNonlinear.png'

# Reading images and reference
preChangeImage = np.array(Image.open(preChangeDataPath))
postChangeImage = np.array(Image.open(postChangeDataPath))
referenceImageTransformed = 1 - np.array(Image.open(referencePath))
print(preChangeImage.shape, postChangeImage.shape, referenceImageTransformed.shape)

# Pre-process/normalize the images
percentileToSaturate = 1
preChangeImage = saturateSomePercentileBandwise(preChangeImage, percentileToSaturate)
postChangeImage = saturateSomePercentileBandwise(postChangeImage, percentileToSaturate)

# Number of spectral bands
numSpectralBands = preChangeImage.shape[2]

# Getting normalized CD map (magnitude map)
detectedChangeMapNormalized, timeVector1FeatureAggregated, timeVector2FeatureAggregated = deepPriorCd(preChangeImage, postChangeImage, manualSeed, outputLayerNumbers)

# Saving features for visualization
# absoluteModifiedTimeVectorDifference=np.absolute(timeVector1FeatureAggregated-timeVector2FeatureAggregated)
# print(absoluteModifiedTimeVectorDifference.shape)
# for featureIter in range(absoluteModifiedTimeVectorDifference.shape[2]):
#    detectedChangeMapThisFeature=absoluteModifiedTimeVectorDifference[:,:,featureIter]
#    detectedChangeMapNormalizedThisFeature=(detectedChangeMapThisFeature-np.amin(detectedChangeMapThisFeature))/(np.amax(detectedChangeMapThisFeature)-np.amin(detectedChangeMapThisFeature))
#    detectedChangeMapNormalizedThisFeature=scaleContrast(detectedChangeMapNormalizedThisFeature)
#    plt.imsave('./savedFeatures/santaBarbara'+'FeatureBest'+str(featureIter)+'.png',np.repeat(np.expand_dims(detectedChangeMapNormalizedThisFeature,2),3,2))


# Getting CD map from normalized CD maps


otsuThreshold = filters.threshold_otsu(detectedChangeMapNormalized)
cdMap = detectedChangeMapNormalized > otsuThreshold
cdMap = morphology.binary_erosion(cdMap)
cdMap = morphology.binary_dilation(cdMap)

# cv2.imwrite(resultPath,((1-cdMap)*255).astype('uint8'))
Image.fromarray(((1 - cdMap) * 255).astype(np.uint8)).save(resultPath)

# Computing quantitative indices
referenceImageTo1DArray = (referenceImageTransformed).ravel()
cdMapTo1DArray = cdMap.astype(int).ravel()
confusionMatrixEstimated = confusion_matrix(y_true=referenceImageTo1DArray, y_pred=cdMapTo1DArray, labels=[0, 1])

tn, fp, fn, tp = confusionMatrixEstimated.ravel()
acc = (tn + tp) / (tn + tp + fp + fn)  # Accuracy (all correct / all)
precision = tp / (tp + fp)  # Precision (true positives / predicted positives)
recall = tp / (tp + fn)  # Sensitivity aka Recall (true positives / all actual positives)
fpr = fp / (fp + tn)  # False Positive Rate (Type I error)
spec = tn / (tn + fp)  # Specificity (true negatives / all actual negatives)
error = (fn + fp) / (tn + tp + fp + fn)  # Misclassification (all incorrect / all)
f1 = (2 * precision * recall) / (precision + recall)
jacc = tp / (tp + fp + fn)  # https://www.mathworks.com/help/images/ref/jaccard.html
dice = 2 * tp / (2 * tp + fp + fn)  # https://www.mathworks.com/help/images/ref/dice.html

metrics = {
    'acc': acc if not np.isnan(acc) else 0,
    'precision': precision if not np.isnan(precision) else 0,
    'recall': recall if not np.isnan(recall) else 0,
    'fpr': fpr if not np.isnan(fpr) else 0,
    'spec': spec if not np.isnan(spec) else 0,
    'error': error if not np.isnan(error) else 0,
    'f1': f1 if not np.isnan(f1) else 0,
    'jacc': jacc if not np.isnan(jacc) else 0,
    'dice': dice if not np.isnan(dice) else 0,
}
print(metrics)
