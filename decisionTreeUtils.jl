include("decisionTreeTypes.jl")

function enumerateLeafs(tree::DecisionTree, input, inputRules = [(eltype(V) == Int64) ? Set(V) : (minimum(V), maximum(V)) for V in input]) 
	if typeof(tree) <: Union{DecisionLeaf, DecisionTestTrainLeaf}
		(inputRules, length(tree.ind), tree.prediction, tree.err)
	elseif typeof(tree) <: Union{DecisionLabelNode, DecisionTestTrainLabelNode}
		splitCol = tree.splitCol
		leftPartition = tree.leftPartition
		labelSet = inputRules[splitCol]
		leftInputRules = [(i == splitCol) ? leftPartition : inputRules[i] for i in 1:length(inputRules)]
		rightInputRules = [(i == splitCol) ? setdiff(labelSet, leftPartition) : inputRules[i] for i in 1:length(inputRules)]
		vcat(enumerateLeafs(tree.left, input, leftInputRules), enumerateLeafs(tree.right, input, rightInputRules))
	else
		splitCol = tree.splitCol
		splitPoint = tree.splitPoint
		varRange = inputRules[splitCol]
		leftInputRules = [(i == splitCol) ? (varRange[1], splitPoint) : inputRules[i] for i in 1:length(inputRules)]
		rightInputRules = [(i == splitCol) ? (splitPoint, varRange[2]) : inputRules[i] for i in 1:length(inputRules)]
		vcat(enumerateLeafs(tree.left, input, leftInputRules), enumerateLeafs(tree.right, input, rightInputRules))
	end
end

function sortSample(sortedInd::AbstractVector{Int64}, selectedInd::AbstractVector{Int64})
	l1 = length(sortedInd)
	l2 = length(selectedInd)
	tmp = fill(false, l1)
	tmp[selectedInd] .= true
	view(sortedInd, find(view(tmp, sortedInd)))
end

function sortReplacementSample(sortedInd::AbstractVector{Int64}, selectedInd::AbstractVector{Int64})
#unlike in the first case selectedInd could be longer and sortedInd and have duplicate values	
	l1 = length(sortedInd)
	l2 = length(selectedInd)
	tmp = zeros(Int64, l1) # fill(false, l1)
	for i in selectedInd
		tmp[i] += 1
	end
	
	out = Vector{Int64}(undef, l2)
	i = 1
	for j in 1:l1
		n = tmp[sortedInd[j]]
		if n >= 1
			out[i:i+n-1] .= sortedInd[j]
		end
		i += n
	end
	return out
	# reduce(vcat, [fill(sortedInd[i], tmp[sortedInd[i]]) for i = 1:l1])
	
	# view(sortedInd, find(view(tmp, sortedInd)))
end

# view(selectedInd, sortperm(view(input, selectedInd)))

# function getSplits(selectedInd::AbstractArray{Bool, 1}, sortInd::AbstractArray{Int64, 1}, invSortInd::AbstractArray{Int64, 1}, splitInd::Int64)
# #convert the original selectedInd array into the two split arrays using the sorting vector and the 
# #split point
# 	rawSplit1 = [selectedInd[sortInd[1:splitInd]]; fill(false, length(selectedInd) - splitInd)]
# 	rawSplit2 = [fill(false, splitInd); selectedInd[sortInd[splitInd+1:end]]]
# 	(view(rawSplit1, invSortInd), view(rawSplit2, invSortInd))
# end

function getSplits(sortedSelection::AbstractVector{Int64}, splitInd::Int64, l::Int64)
#convert the original selectedInd array into the two split arrays using the sorting vector and the 
#split point
	out1 = fill(false, l)
	out2 = fill(false, l)
	if splitInd > 0
		# out1[view(sortedSelection, 1:splitInd)] = true
		out1[sortedSelection[1:splitInd]] .= true
	end

	if splitInd < length(sortedSelection)
		# out2[view(sortedSelection, splitInd+1:length(sortedSelection))] = true
		out2[sortedSelection[splitInd+1:length(sortedSelection)]] .= true
	end
	(out1, out2)
end

function getSplits(selectedInd::AbstractVector{Bool}, sortedInd::AbstractVector{Int64}, splitInd::Int64)
#convert the original selectedInd array into the two split arrays using the sorting vector and the 
#split point
	l = length(selectedInd)
	out1 = fill(false, l)
	out2 = fill(false, l)
	
	if splitInd >= 1
		@inbounds @simd for i in 1:splitInd
			out1[sortedInd[i]] = selectedInd[sortedInd[i]]
		end
	end

	if splitInd<l
		@inbounds @simd for i in splitInd+1:l
			out2[sortedInd[i]] = selectedInd[sortedInd[i]]
		end
	end

	(out1, out2)
end

function indToBool(inds::AbstractVector{Int64}, l::Int64)
	out = fill(false, l)
	if !isempty(inds)
		out[inds] = true
	end
	out 
end

function selectSum(V::Vector{T}, ind::Vector{Bool}) where T <: Real
	s = zero(T)
	@inbounds @fastmath @simd for i in eachindex(V)
		s += ifelse(ind[i], V[i], zero(T))
		# if ind[i]
		# 	s+= V[i]
		# end
	end

	# for i in eachindex(V)
	# 	if ind[i]
	# 		s+=V[i]
	# 	end
	# end
	return s
end

#---------------Error Calc Functions---------------------------------
function getTreeError(tree::DecisionTree)
	if typeof(tree) <: Union{DecisionLeaf, DecisionTestTrainLeaf}
		[tree.err*length(tree.ind) length(tree.ind)]
	else
		getTreeError(tree.left) .+ getTreeError(tree.right)
	end
end

function getTreeTestError(tree::DecisionTestTrainTree)
	if typeof(tree) <: DecisionTestTrainLeaf
		[length(tree.testInd) == 0 ? 0.0 : tree.testErr*length(tree.testInd) length(tree.testInd)]
	else
		getTreeTestError(tree.left) .+ getTreeTestError(tree.right)
	end
end

function calcTreeTestError(tree::DecisionTestTrainTree)
	out = getTreeTestError(tree)
	out[1]/out[2]
end

function getTreePredictions(tree::DecisionTree)
	if typeof(tree) <: Union{DecisionLeaf, DecisionTestTrainLeaf}
		[(tree.prediction, tree.ind)]
	else
		vcat(getTreePredictions(tree.left), getTreePredictions(tree.right))
	end
end

function getTestTrainTreePredictions(tree::DecisionTestTrainTree)
	if typeof(tree) <: DecisionTestTrainLeaf
		[(tree.prediction, tree.ind, tree.testInd)]
	else
		vcat(getTestTrainTreePredictions(tree.left), getTestTrainTreePredictions(tree.right))
	end
end

function formOrderedTreeOutput(tree::DecisionTree)
	predictions = getTreePredictions(tree)
	indList = mapreduce(a -> a[2], vcat, predictions)
	output = Vector{Float64}(undef, length(indList))
	for i in 1:length(predictions)
		output[predictions[i][2]] .= predictions[i][1]
	end
	return output
end

function formOrderedTestTrainTreeOutput(tree::DecisionTestTrainTree)
	predictions = getTestTrainTreePredictions(tree)
	trainIndList = mapreduce(a -> a[2], vcat, predictions)
	testIndList = mapreduce(a -> a[3], vcat, predictions)
	output = Vector{Float64}(undef, length(trainIndList))
	testOutput = Vector{Float64}(undef, length(testIndList))
	
	for i in 1:length(predictions)
		output[predictions[i][2]] .= predictions[i][1]
		testOutput[predictions[i][3]] .= predictions[i][1]
	end
	
	return (output, testOutput)
end

function getTreeBoolError(tree::DecisionTree)
#measures classification error of a tree, only makes sense for a boolean output set
	if typeof(tree) <: DecisionLeaf
		pred = tree.prediction
		[min(pred, 1-pred)*length(tree.ind) length(tree.ind)]
	else
		getTreeBoolError(tree.left) .+ getTreeBoolError(tree.right)
	end
end

function calcTreeError(tree::DecisionTree)
	out = getTreeError(tree)
	out[1]/out[2]
end

function calcTreeBoolError(tree::DecisionTree)
	out = getTreeBoolError(tree)
	out[1]/out[2]
end

function calcClassErr(pred::outputBoolType, out::outputBoolType)
	1-mean(pred.==out)
end

function calcOutputErr(pred::outputBoolType, out::outputBoolType)
	1-mean(pred.==out)
end

function calcOutputErr(pred::outputNumType, out::outputNumType)
	mean((pred.-out).^2)
end

function calcMultiErrs(multiOut::Vector{U}, output::T) where U <: outputNumType where T <: Union{outputBoolType, outputNumType}
	combinedForestSumOut = zeros(length(output[1]))
	combinedForestMultiSums = [combinedForestSumOut = combinedForestSumOut .+ a for a in multiOut]
	forestErrs = if T <: outputBoolType
		[calcOutputErr(combinedForestMultiSums[j]/j .> 0.5, output) for j in 1:length(multiOut)]
	else
		[calcOutputErr(combinedForestMultiSums[j]/j, output) for j in 1:length(multiOut)]
	end
end

function calcBoolEntropy(V::T) where T <: AbstractArray{Bool, 1}
#calculates information entropy of boolean output vector
	n_true = sum(V)
	l = length(V)
	p = n_true/l
	if (p == 1) | (p == 0)
		0
	else
		-p*log2(p) - (1-p)*log2(1-p)
	end
end 

function calcBoolError(V::T, pred::Bool) where T <: AbstractArray{Bool, 1}
#calculates classification error for a boolean vector V where the predicted value is pred
	if pred
		(length(V) - sum(V))/length(V)
	else
		sum(V)/length(V)
	end
end

#-------------------------Prediction Functions-------------------------------------------
#convert boolean value to left or right with the convention of true meaning left
getDir(b::Bool) = b ? :left : :right

#at any graph terminus extract prediction
predictGraph(graph::DecisionGraphTerminus, input) = graph.prediction

#conditions for choosing split for regular nodes and label nodes for single value inputs
chooseSide(graph::LabelBranch, input::Real) = in(input, graph.leftPartition) 
chooseSide(graph::ValueBranch, input::Real) = (input <= graph.splitPoint) 
chooseSide(graph::DecisionBranch, input::Missing) = graph.missingLeft

#conditions for choosing split for regular nodes and label nodes for single vector inputs
chooseSide(graph::Union{DecisionGraph, DecisionTree}, input::singleInputType) = chooseSide(graph, input[graph.splitCol])
#conditions for choosing split for regular nodes and label nodes for vectors of column inputs
chooseSide(graph::DecisionBranch, input, i::Int64) = chooseSide(graph, input[graph.splitCol][i])

#Extract left or right branch based on condition from input
getSide(graph::Branch, input) = getfield(graph, getDir(chooseSide(graph, input)))
getSide(graph::DecisionBranch, input, i::Int64) = getfield(graph, getDir(chooseSide(graph, input, i)))

#branch left or right tree branch depending on condition
predictGraph(graph::DecisionGraph, input) = predictGraph(getSide(graph, input), input)

#at any tree leaf extract prediction, number of examples, and depth 
predictTree(graph::DecisionLeaf, input, r = 1) = (graph.prediction, length(graph.ind), r)

#branch left or right tree branch depending on condition and add 1 to depth
predictTree(graph::DecisionTree, input, r = 1) = predictTree(getSide(graph, input), input, r+1)

#branch left or right depending on condition for index i 
predictSimpleTree(graph::DecisionTerm, input::Vector{T}, i::Int64) where T <: Any = Float64(graph.prediction)
predictSimpleTree(graph::DecisionBranch, input::Vector{T}, i::Int64) where T <: Any = predictSimpleTree(getSide(graph, input, i), input, i)

#branch left and right for input i keeping track of predictions and counts along path as well as path itself
predictTree(graph::DecisionTerm, input, i::Int64, path::Vector{Bool}, predictions::Vector{Float64}, counts::Vector{Int64}) = ([predictions; Float64(graph.prediction)], [counts; length(graph.ind)], path)
predictTree(graph::DecisionBranch, input, i::Int64, path::Vector{Bool}, predictions::Vector{Float64}, counts::Vector{Int64}) = predictTree(getSide(graph, input, i), input, i, [path; chooseSide(graph, input, i)], [predictions; Float64(graph.leaf.prediction)], [counts; length(graph.leaf.ind)])

#Get prediction tree output for inputs tracking entire history through tree
function runPredictTree(graph::DecisionTree, input::Vector{T}) where T <: Any
	featureType = typeof(input[1])
	@assert (featureType <: AbstractVector)
	elType = typeof(input[1][1])
	@assert isbits(input[1][1]) "$(input[1][1]) is not a bits type"
	m = length(input)
	l = length(input[1])
	if m > 1
		for i = 2:m
			assert(length(input[i]) == l)
		end
	end
	println()
	t = time()
	outputs = Vector{Vector{Float64}}(undef, l)
	exampleCounts = Vector{Vector{Int64}}(undef, l)
	paths = Vector{Vector{Bool}}(undef, l)
	for i in 1:l
		out = predictTree(graph, input, i, Vector{Bool}(), Vector{Float64}(), Vector{Int64}())
		outputs[i] = out[1]
		exampleCounts[i] = out[2]
		paths[i] = out[3]
		if time() - t > 5
			t = time()
			print("\r\u1b[K\u1b[A")
			print("\r")
			print("\u1b[K")
			println(string("Done with input ", i, " of ", l))
		end
	end
	(outputs, exampleCounts, paths)
end

function runSimplePredictTree(graph::DecisionTree, input::Vector{T}) where T <: Any
	featureType = typeof(input[1])
	@assert (featureType <: AbstractVector)
	@assert isbits(input[1][1]) "$(input[1][1]) is not a bits type"
	m = length(input)
	l = length(input[1])
	if m > 1
		for i = 2:m
			@assert (length(input[i]) == l)
		end
	end

	println()
	t = time()
	output = Vector{Float64}(undef, l)	
	for i in 1:l
		output[i] = predictSimpleTree(graph, input, i)
		if time() - t > 5
			t = time()
			print("\r\u1b[K\u1b[A")
			print("\r")
			print("\u1b[K")
			println(string("Done with input ", i, " of ", l))
		end
	end
	output
end



function predictForest(forest::decisionForest, input::Vector{T}) where T <: Any
	out = [runSimplePredictTree(a[2], input) for a in forest]
	meanOut = reduce((a, b) -> a.+b, out)/length(out)
	stdOut = sqrt.(mapreduce(a -> (a.-meanOut).^2, (a, b) -> a.+b, out)/length(out))
	(meanOut, stdOut)
end

#--------------------------------Conversion Functions-----------------------------------------
convTreeGraph(tree::DecisionLeaf) = DecisionGraphTerminus(tree.prediction)
convTreeGraph(tree::DecisionNode) = DecisionGraphNode(tree.leaf.prediction, tree.splitCol, tree.splitPoint, convTreeGraph(tree.left), convTreeGraph(tree.right))
convTreeGraph(tree::DecisionLabelNode) = DecisionGraphLabelNode(tree.leaf.prediction, tree.splitCol, tree.leftPartition, convTreeGraph(tree.left), convTreeGraph(tree.right))