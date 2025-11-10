--------------------------------------------------------------------------------
-- Iris model specification (Iris Setosa / Versicolor / Virginica)
-- Inputs: sepal length, sepal width, petal length, petal width (cm)
-- Outputs: 3 scores, one per class

--------------------------------------------------------------------------------
-- Inputs

type Input = Tensor Real [4]

-- Feature indices
sepalLength = 0
sepalWidth  = 1
petalLength = 2
petalWidth  = 3

-- Valid (problem-space) ranges for the Iris dataset (approx. extrema)
type UnnormalisedInput = Tensor Real [4]

minimumInputValues : UnnormalisedInput
minimumInputValues = [4.3, 2.0, 1.0, 0.1]

maximumInputValues : UnnormalisedInput
maximumInputValues = [7.9, 4.4, 6.9, 2.5]

validInput : UnnormalisedInput -> Bool
validInput x = forall i .
  minimumInputValues ! i <= x ! i <= maximumInputValues ! i

--------------------------------------------------------------------------------
-- Outputs

type Output = Tensor Real [3]

-- Class indices
setosa     = 0
versicolor = 1
virginica  = 2

--------------------------------------------------------------------------------
-- The network

@network
iris : Input -> Output

-- Assume the exported ONNX includes any scaling inside the graph.
-- If not, you can insert a normalisation step like in the ACAS example.
applyIris : UnnormalisedInput -> Output
applyIris x = iris x

-- "advises i x" means class i has the strictly minimal score for input x
advises : Index 3 -> UnnormalisedInput -> Bool
advises i x = forall j . i != j => applyIris x ! i < applyIris x ! j

--------------------------------------------------------------------------------
-- Property I1: Very small petals -> Setosa
-- (Classic separation: Setosa has distinctly tiny petals.)
smallPetals_Setosa : UnnormalisedInput -> Bool
smallPetals_Setosa x =
  x ! petalLength <= 2.0 and
  x ! petalWidth  <= 0.6

@property
propertyI1_Setosa : Bool
propertyI1_Setosa = forall x .
  validInput x and smallPetals_Setosa x =>
  advises setosa x

--------------------------------------------------------------------------------
-- Property I2: Very large petals -> Virginica
-- (Safe interior region of Virginica cluster.)
largePetals_Virginica : UnnormalisedInput -> Bool
largePetals_Virginica x =
  x ! petalLength >= 5.5 and
  x ! petalWidth  >= 1.9

@property
propertyI2_Virginica : Bool
propertyI2_Virginica = forall x .
  validInput x and largePetals_Virginica x =>
  advises virginica x

--------------------------------------------------------------------------------
-- Property I3: Mid-range petals -> Versicolor
-- (A conservative interior band for Versicolor.)
midPetals_Versicolor : UnnormalisedInput -> Bool
midPetals_Versicolor x =
  3.2 <= x ! petalLength <= 4.8 and
  1.0 <= x ! petalWidth  <= 1.7

@property
propertyI3_Versicolor : Bool
propertyI3_Versicolor = forall x .
  validInput x and midPetals_Versicolor x =>
  advises versicolor x
