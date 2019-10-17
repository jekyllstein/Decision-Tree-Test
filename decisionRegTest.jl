include("decisionTreeExperimentv2.jl")

using Plots
plotly()

N = 10000

X = rand(N)

Y = [begin
	X[i]^2 + (rand() - 0.5)/5
end
for i in 1:length(X)]

Ytest = X.^2

testInput = [begin
	[X[i]]
end
for i in 1:length(X)]

input = [X]

# sortedInd = [sortperm(v) for v in input]
sortedInd = [Vector{Int64}() for v in input]
@time tree = makeDecisionTree(input, Y, Y.^2);	
# println()

# Profile.clear()
# @profile tree = makeDecisionTree([X], Y, Y.^2);
# Profile.print()

# Profile.clear()
# @profile tree = makeDecisionTree([X], Y, Y.^2);
# Profile.print()

# testPredict = [predictTree(tree, v)[1] for v in testInput]
# origTrainErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Y)
# origTestErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Ytest)
# (prunedTree, err) = pruneDecisionTree(tree, testInput, Ytest)
# prunedTreeOutput = [predictTree(prunedTree, v)[1] for v in testInput]
# treeOutput = [predictTree(tree, v)[1] for v in testInput]

# scatter(X, Y, markersize=0.2)
# scatter(X, treeOutput, markersize=0.2)
# scatter(X, prunedTreeOutput, markersize=0.2)
# println()

(unprunedForest, unprunedForestTrainErrs, unprunedForestTestErrs, bestInd1, unprunedForestTrainPredict, unprunedForestTestPredict) = makeRandomForest([X], Y, testInput, Ytest, B = 100, prune=false);
(prunedForest, forestTrainErrs, forestTestErrs, bestInd2, prunedForestTrainPredict, prunedForestTestPredict) = makeRandomForest([X], Y, testInput, Ytest, B = 100);
prunedForestPredict = [predictForest(prunedForest, v)[1] for v in testInput]
prunedForestTrainErr = calcOutputErr(prunedForestPredict, Y)
prunedForestTestErr = calcOutputErr(prunedForestPredict, Ytest) #0.2317

scatter(X, prunedForestTestPredict, markersize=0.2)
scatter(X, unprunedForestTestPredict, markersize=0.2)

# X1 = ceil.(Int64, 3*rand(N))
# X2 = rand(N)

# Y = [begin
# 	X2[i].^(X1[i]/2) + (rand() - 0.5)/5
# end
# for i in 1:length(X1)]

# Ytest = X2.^(X1/2)

# input = Vector{Vector{Real}}(2) 
# input[1] = X1
# input[2] = X2

# testInput = [begin
# 	tmp = Array{Real, 1}(2)
# 	tmp[1] = X1[i]
# 	tmp[2] = X2[i]
# 	tmp
# end
# for i in 1:length(X1)]

# @time tree = makeDecisionTree(input, Y, Y.^2);

# Profile.clear()
# @profile tree = makeDecisionTree(input, Y, Y.^2);	
# # Profile.print()

# Profile.clear()
# @profile tree = makeDecisionTree(input, Y, Y.^2);	
# Profile.print(format=:flat)
# println()
# testPredict = [predictTree(tree, v)[1] for v in testInput]
# origTrainErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Y)
# origTestErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Ytest)
# (prunedTree, err) = pruneDecisionTree(tree, testInput, Ytest)
# prunedTreeOutput = [predictTree(prunedTree, v)[1] for v in testInput]
# treeOutput = [predictTree(tree, v)[1] for v in testInput]

# scatter(X2, Y, markersize=0.2)
# scatter(X2, treeOutput, markersize=0.2)
# scatter(X2, prunedTreeOutput, markersize=0.2)

# forest = makeRandomForest(input, Y)
# forestPredict = [predictForest(forest, v)[1] for v in testInput]
# forestTrainErr = calcOutputErr(forestPredict, Y)
# forestTestErr = calcOutputErr(forestPredict, Ytest)

# (unprunedForest, unprunedForestTrainErrs, unprunedForestTestErrs, bestInd1, unprunedForestTrainPredict, unprunedForestTestPredict) = makeRandomForest(input, Y, testInput, Ytest, B = 100, prune=false)
# (prunedForest, forestTrainErrs, forestTestErrs, bestInd2, prunedForestTrainPredict, prunedForestTestPredict) = makeRandomForest(input, Y, testInput, Ytest, B = 1000)
# prunedForestPredict = [predictForest(prunedForest, v)[1] for v in testInput]
# prunedForestTrainErr = calcOutputErr(prunedForestPredict, Y)
# prunedForestTestErr = calcOutputErr(prunedForestPredict, Ytest) #0.2317


# X1 = rand(Bool, N)
# X2 = rand(N)

# Y = [begin
# 	(X1[i] ? X2[i].^2 : X2[i].^(1/2)) + (rand() - 0.5)/5
# end
# for i in 1:length(X1)]

# Ytest = [begin
# 	(X1[i] ? X2[i].^2 : X2[i].^(1/2))
# end
# for i in 1:length(X1)]

# input = Vector{Vector{Real}}(2) 
# input[1] = X1
# input[2] = X2

# testInput = [begin
# 	tmp = Array{Real, 1}(2)
# 	tmp[1] = X1[i]
# 	tmp[2] = X2[i]
# 	tmp
# end
# for i in 1:length(X1)]

# @time tree = makeDecisionTree(input, Y, Y.^2);

# # Profile.clear()
# # @profile tree = makeDecisionTree(input, Y, Y.^2);	
# # # Profile.print()

# Profile.clear()
# @profile tree = makeDecisionTree(input, Y, Y.^2);	
# Profile.print(format=:flat)
# # println()
# testPredict = [predictTree(tree, v)[1] for v in testInput]
# origTrainErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Y)
# origTestErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Ytest)
# (prunedTree, err) = pruneDecisionTree(tree, testInput, Ytest)
# prunedTreeOutput = [predictTree(prunedTree, v)[1] for v in testInput]
# treeOutput = [predictTree(tree, v)[1] for v in testInput]

# scatter(X2, Y, markersize=0.2)
# scatter(X2, treeOutput, markersize=0.2)
# scatter(X2, prunedTreeOutput, markersize=0.2)



X1 = rand(Bool, N)
X2 = rand(N)

Y = [begin
	# X2[i]^(2/X1[i]) + (rand()-0.5)/5
	(X1[i] == 1 ? X2[i].^2 : X2[i].^(1/2)) + (rand() - 0.5)/5
end
for i in 1:length(X1)]

Ytest = [begin
	# X2[i]^(2/X1[i])

	(X1[i] == 1 ? X2[i].^2 : X2[i].^(1/2))
end
for i in 1:length(X1)]

# input = Vector{Vector{Real}}(2) 
input = Vector{Any}(2) 
input[1] = X1
input[2] = X2

testInput = [begin
	tmp = Array{Real, 1}(2)
	tmp[1] = X1[i]
	tmp[2] = X2[i]
	tmp
end
for i in 1:length(X1)]

@time tree = makeDecisionTree(input, Y, Y.^2);
# @time tree = makeDecisionTree([X1, X2], Y, Y.^2);

# Profile.clear()
# @profile tree = makeDecisionTree(input, Y, Y.^2);	
# # Profile.print()

Profile.clear()
@profile tree = makeDecisionTree(input, Y, Y.^2);	
Profile.print(format=:flat)
# # println()
testPredict = [predictTree(tree, v)[1] for v in testInput]
origTrainErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Y)
origTestErr = calcOutputErr([predictTree(tree, v)[1] for v in testInput], Ytest)
(prunedTree, err) = pruneDecisionTree(tree, testInput, Ytest)
prunedTreeOutput = [predictTree(prunedTree, v)[1] for v in testInput]
treeOutput = [predictTree(tree, v)[1] for v in testInput]

scatter(X2, Y, markersize=0.2)
scatter(X2, treeOutput, markersize=0.2)
scatter(X2, prunedTreeOutput, markersize=0.2)

(unprunedForest, unprunedForestTrainErrs, unprunedForestTestErrs, bestInd1, unprunedForestTrainPredict, unprunedForestTestPredict) = makeRandomForest(input, Y, testInput, Ytest, B = 100, prune=false);
(prunedForest, forestTrainErrs, forestTestErrs, bestInd2, prunedForestTrainPredict, prunedForestTestPredict) = makeRandomForest(input, Y, testInput, Ytest, B = 100);
prunedForestPredict = [predictForest(prunedForest, v)[1] for v in testInput]
prunedForestTrainErr = calcOutputErr(prunedForestPredict, Y)
prunedForestTestErr = calcOutputErr(prunedForestPredict, Ytest) #0.2317

scatter(X2, prunedForestTestPredict, markersize=0.2)
scatter(X2, unprunedForestTestPredict, markersize=0.2)