--------------------------------------------------------------------------------
-- Full Vehicle Specification: MNIST Robustness and Strong Robustness

-- This file is based on the Vehicle tutorial for image classification networks.
-- It includes the definition for Strong Classification Robustness.
--------------------------------------------------------------------------------
-- Inputs and outputs

-- Define the type for our input images
type Image = Tensor Real [28, 28]

-- The type of the output labels
-- i.e a number between 0 and 9, one for each digit
type Label = Index 10

-- A predicate that states that all the pixel values in a given image are
-- in the range 0.0 to 1.0
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

--------------------------------------------------------------------------------
-- Network

-- Declare the network used to classify images. The output of the network is a
-- score for each of the digits 0 to 9.
@network
classifier : Image -> Tensor Real [10]

-- The classifier advises that input image `x` has label `i` if the score
-- for label `i` is greater than the score of any other label `j`.
advises : Image -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Definition of robustness around a point

-- Parameter for the radius of the perturbation ball (L-infinity norm).
@parameter
epsilon : Real

-- NEW PARAMETERS FOR STRONG CLASSIFICATION ROBUSTNESS (Slightly adjusted for typical use)
-- eta: The score threshold that the WRONG label's score must stay below.
@parameter
eta : Real

-- wrongLabel: The specific index of the WRONG label to check.
@parameter
wrongLabel : Label

-- Next we define what it means for an image `x` to be in a ball of
-- size epsilon around 0.
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

-- The original robustness definition: the label is maintained.
robustAround : Image -> Label -> Bool
robustAround image label = forall pertubation .
  let perturbedImage = image - pertubation in
  boundedByEpsilon pertubation and validImage perturbedImage =>
    advises perturbedImage label

-- NEW DEFINITION: STRONG CLASSIFICATION ROBUSTNESS
-- Requires the score of a specific WRONG label (`wrongLabel`) to be below a threshold (`eta`)
-- throughout the epsilon ball around the image.
strongRobustAround : Image -> Label -> Bool
strongRobustAround image label = forall pertubation .
  let perturbedImage = image - pertubation in
  boundedByEpsilon pertubation and validImage perturbedImage =>
    -- Requirement: The score of the designated wrong class must be <= eta
    classifier perturbedImage ! wrongLabel <= eta

--------------------------------------------------------------------------------
-- Robustness with respect to a dataset

-- Parameter for the size of the training dataset.
@parameter(infer=True)
n : Nat

-- Datasets
@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

-- The original robustness property (checked over the entire dataset).
@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)

-- THE REQUIRED STRONG CLASSIFICATION ROBUSTNESS PROPERTY
@property
strong_classification_robustness_property : Vector Bool n
strong_classification_robustness_property = foreach i .
  strongRobustAround (trainingImages ! i) (trainingLabels ! i)