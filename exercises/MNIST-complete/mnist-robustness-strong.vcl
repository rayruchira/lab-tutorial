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

-- "advises" (same as your example, kept for reference/use if needed)
advises : Image -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Parameters

@parameter
epsilon : Real            -- radius of the L∞ ball

@parameter
eta : Real                -- SCR upper bound on non-true class scores

--------------------------------------------------------------------------------
-- Helpers

-- L∞-ball around 0 with radius epsilon
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

--------------------------------------------------------------------------------
-- Strong Classification Robustness around a point
-- For every perturbation within epsilon (and still a valid image),
-- ALL non-true classes' scores must be ≤ eta.

strongAround : Image -> Label -> Bool
strongAround image label =
  forall perturbation .
    let x' = image - perturbation in
    boundedByEpsilon perturbation and validImage x' =>
      (forall i . i != label => classifier x' ! i <= eta)

--------------------------------------------------------------------------------
-- Dataset-level property (like your previous spec)

@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
strongRobust : Vector Bool n
strongRobust = foreach k . strongAround (trainingImages ! k) (trainingLabels ! k)
