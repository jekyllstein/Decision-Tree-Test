include("decisionTreeExperimentv2.jl")

N = 10000

X1 = ceil.(Int64, 10*rand(N))
X2 = rand(N)

Y = [begin
	if in(X1[i], [1, 3, 5, 7, 9])
    	X2[i] < 0.5 + (rand() - 0.5)/2
    else
    	X2[i] > 0.5 + (rand() - 0.5)/2 
    end
end
for i in 1:length(X1)]

Ytest = [begin
	if in(X1[i], [1, 3, 5, 7, 9])
    	X2[i] < 0.5
    else
    	X2[i] > 0.5
    end
end
for i in 1:length(X1)]

tree = makeDecisionTree([X1, X2], Y)

testInput = [begin
	tmp = Array{Real, 1}(2)
	tmp[1] = X1[i]
	tmp[2] = X2[i]
	tmp
end
for i in 1:length(X1)]

testPredict = [predictTree(tree, v)[1] for v in testInput]
origTrainErr = calcClassErr([predictTree(tree, v)[1] for v in testInput] .> 0.5, Y)
origTestErr = calcClassErr([predictTree(tree, v)[1] for v in testInput] .> 0.5, Ytest)
(prunedTree, err) = pruneDecisionTree(tree, testInput, Ytest)

forest = makeRandomForest([X1, X2], Y)
forestPredict = [predictForest(forest, v)[1] for v in testInput]
forestTrainErr = calcClassErr(forestPredict .> 0.5, Y)
forestTestErr = calcClassErr(forestPredict .> 0.5, Ytest)

(unprunedForest, unprunedForestErrs, bestInd1, unprunedForestPredict) = makeRandomForest([X1, X2], Y, testInput, Ytest, B = 1000, prune=false)
(prunedForest, forestErrs, bestInd2, prunedForestPredict) = makeRandomForest([X1, X2], Y, testInput, Ytest, B = 100)
prunedForestPredict = [predictForest(prunedForest, v)[1] for v in testInput]
prunedForestTrainErr = calcClassErr(prunedForestPredict .> 0.5, Y)
prunedForestTestErr = calcClassErr(prunedForestPredict .> 0.5, Ytest) #0.2317