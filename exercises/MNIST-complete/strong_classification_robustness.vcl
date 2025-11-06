--------------------------------------------------------------------------------
-- Inputs and outputs

type Image = Tensor Real [28, 28]
type Label = Index 10

validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

--------------------------------------------------------------------------------
-- Network

@network
classifier : Image -> Tensor Real [10]

advises : Image -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Parameters

@parameter
epsilon : Real

@parameter
eta : Real

--------------------------------------------------------------------------------
-- Helpers

boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

--------------------------------------------------------------------------------
-- Strong Classification Robustness around a point

strongAround : Image -> Label -> Bool
strongAround image label =
  forall perturbation .
    let xPrime = image - perturbation in
    boundedByEpsilon perturbation and validImage xPrime =>
      (forall i . i != label => classifier xPrime ! i <= eta)

--------------------------------------------------------------------------------
-- Dataset-level property

@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
strongRobust : Vector Bool n
strongRobust = foreach k . strongAround (trainingImages ! k) (trainingLabels ! k)
