include("decisionTreeExperimentv3.jl")

N = 1000000

X1 = rand(N)
X2 = rand(N)
output = X1.^2 .+ X2.^2
sqOutput = output.^2

input = [X1, X2]

tree = makeDecisionTree(input, output, sqOutput, p = 100000);
@time tree = makeDecisionTree(input, output, sqOutput, p = 1000);