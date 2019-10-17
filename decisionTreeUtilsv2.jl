import Base.length

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
	tmp[selectedInd] = true
	view(sortedInd, find(view(tmp, sortedInd)))
end

function length(gen::Base.Generator)
	c = 0
	for i in gen
		if (typeof(c) <: Base.Generator)
			c+=length(i)
		else
			c+=1
		end
	end
	return c
end

# function length(gen::Base.Iterators)
# 	c = 0
# 	for i in gen
# 		for j in i
# 			c+=1
# 		end
# 	end
# 	return c
# end


function sortReplacementSample(sortedInd::AbstractVector{Int64}, selectedInd, tmp::Vector{Int64})
#unlike in the first case selectedInd could be longer and sortedInd and have duplicate values
	l1 = length(sortedInd)
	l2 = length(selectedInd)
	# tmp = zeros(Int64, l1) # fill(false, l1)
	# @inbounds @simd for i in 1:l1
	# 	tmp[i] = 0
	# end
	fill!(tmp, 0)
	for i in selectedInd
		
			tmp[i] += 1
		
		

	end
	# println(sum(tmp))
	# out = Vector{Int64}(l2)
	i = 1
	for j in 1:l1
		n = tmp[sortedInd[j]]
		if n >= 1
			selectedInd[i:i+n-1] .= sortedInd[j]
		end
		# tmp[sortedInd[j]] = 0
		i += n
	end
	return selectedInd
	# (sortedInd[j] for j in 1:l1 for i in 1:tmp[sortedInd[j]])
	# reduce(vcat, [fill(sortedInd[i], tmp[sortedInd[i]]) for i = 1:l1])
	
	# view(sortedInd, find(view(tmp, sortedInd)))
end

function presortTest(N::Int64, l::Int64)
	V = rand(N)
	sortedInd = sortperm(V)
	selectedInd = shuffle(1:N)[1:l]
	tmp = Vector{Int64}(N)

	@timed view(selectedInd, sortperm(view(V, selectedInd)))
	_, t1 = @timed view(selectedInd, sortperm(view(V, selectedInd)))
	@timed sortReplacementSample(sortedInd, selectedInd, tmp)
	_, t2 = @timed sortReplacementSample(sortedInd, selectedInd, tmp)
	[N, l, t1, t2, t1>t2]
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
		out1[sortedSelection[1:splitInd]] = true
	end

	if splitInd < length(sortedSelection)
		# out2[view(sortedSelection, splitInd+1:length(sortedSelection))] = true
		out2[sortedSelection[splitInd+1:length(sortedSelection)]] = true
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

function selectedSum(V::Vector, ind::AbstractArray)
	s = zero(eltype(V))
	# println(string("ind = ", ind))
	# println(string("type of ind = ", typeof(ind)))
	@inbounds for i in ind
		
			s+= V[i]
		
	end
	return s
end

function selectedSum(f, inds::AbstractArray, V::AbstractArray)
	s = zero(eltype(V))
	@inbounds for i in inds
			s+= f(V[i])
	end
	return s
end

function selectedSum(V::Vector, ind::Base.Generator)
	s = zero(eltype(V))
	# println(string("ind = ", ind))
	# println(string("type of ind = ", typeof(ind)))
	# println(collect(ind))
	for i in collect(ind)
		# println("i = ", collect(i));
		if !(typeof(i) <: Base.Generator)
			# println(i)
			s+= V[i]
		else
			for j in collect(i)
				# println(j)
				s+=V[j]
			end
		end
	end
	return s
end

#---------------Error Calc Functions---------------------------------
function getTreeError(tree::DecisionTree)
	if typeof(tree) <: Union{DecisionLeaf, DecisionTestTrainLeaf}
		[tree.err*length(tree.ind) length(tree.ind)]
	else
		getTreeError(tree.left) + getTreeError(tree.right)
	end
end

function getTreeTestError(tree::DecisionTestTrainTree)
	if typeof(tree) <: DecisionTestTrainLeaf
		[length(tree.testInd) == 0 ? 0.0 : tree.testErr*length(tree.testInd) length(tree.testInd)]
	else
		getTreeTestError(tree.left) + getTreeTestError(tree.right)
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
	output = Vector{Float64}(length(indList))
	for i in 1:length(predictions)
		output[predictions[i][2]] = predictions[i][1]
	end
	return output
end

function formOrderedTestTrainTreeOutput(tree::DecisionTestTrainTree)
	predictions = getTestTrainTreePredictions(tree)
	trainIndList = mapreduce(a -> a[2], vcat, predictions)
	testIndList = mapreduce(a -> a[3], vcat, predictions)
	output = Vector{Float64}(length(trainIndList))
	testOutput = Vector{Float64}(length(testIndList))
	
	for i in 1:length(predictions)
		output[predictions[i][2]] = predictions[i][1]
		testOutput[predictions[i][3]] = predictions[i][1]
	end
	
	return (output, testOutput)
end

function getTreeBoolError(tree::DecisionTree)
#measures classification error of a tree, only makes sense for a boolean output set
	if typeof(tree) <: DecisionLeaf
		pred = tree.prediction
		[min(pred, 1-pred)*length(tree.ind) length(tree.ind)]
	else
		getTreeBoolError(tree.left) + getTreeBoolError(tree.right)
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
function predictGraph(graph::DecisionGraph{T}, input::U) where T <: AbstractFloat where U <: Real
	if typeof(graph) == DecisionGraphTerminus{T}
		graph.prediction
	elseif typeof(graph) == DecisionGraphLabelNode{T}
		in(input, graph.leftPartition) ? predictGraph(graph.left, input) : predictGraph(graph.right, input)
	else
		input <= graph.splitPoint ? predictGraph(graph.left, input) : predictGraph(graph.right, input)
	end
end

function predictTree(graph::DecisionTree{T}, input::U, r = 1) where T <: AbstractFloat where U <: Real
	if typeof(graph) == DecisionLeaf{T}
		(graph.prediction, length(graph.ind), r)
	elseif typeof(graph) == DecisionLabelNode{T}
		in(input, graph.leftPartition) ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
	else
		input <= graph.splitPoint ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
	end
end

function predictTree(graph::DecisionTree, input::T, r = 1)  where T <: Union{U, Missing} where U <: Real
	if typeof(graph) <: DecisionLeaf
		(graph.prediction, length(graph.ind), r)
	elseif typeof(graph) <: DecisionNullLabelNode
		if input.hasvalue
			in(input.value, graph.leftPartition) ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		else
			graph.nullLeft ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		end
	elseif typeof(graph) <: DecisionNullNode
		if input.hasvalue
			input.value <= graph.splitPoint ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		else
			graph.nullLeft ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		end
	else
		error("input contains null values which tree is not designed to handle")
	end
end

function predictGraph(graph::DecisionGraph{T}, input::AbstractVector{U}) where T <: AbstractFloat where U <: Real
	if typeof(graph) == DecisionGraphTerminus{T}
		graph.prediction
	elseif typeof(graph) == DecisionGraphLabelNode{T}
		in(input[graph.splitCol], graph.leftPartition) ? predictGraph(graph.left, input) : predictGraph(graph.right, input)
	else
		input[graph.splitCol] <= graph.splitPoint ? predictGraph(graph.left, input) : predictGraph(graph.right, input)
	end
end

function predictTree(graph::DecisionTree, input::T, r = 1) where T <: DenseArray{U, 1} where U <: Real
	if typeof(graph) <: DecisionLeaf
		(graph.prediction, length(graph.ind), r)
	elseif typeof(graph) <: DecisionLabelNode
		in(input[graph.splitCol], graph.leftPartition) ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
	else
		input[graph.splitCol] <= graph.splitPoint ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
	end
end

function runPredictTree(graph::DecisionTree, input::Vector{T}) where T <: Any
	featureType = typeof(input[1])
	assert(featureType <: AbstractVector)
	elType = typeof(input[1][1])
	assert(isbits(elType))
	m = length(input)
	l = length(input[1])
	if m > 1
		for i = 2:m
			assert(length(input[i]) == l)
		end
	end
	println()
	outputs = Vector{Vector{Float64}}(l)
	exampleCounts = Vector{Vector{Int64}}(l)
	paths = Vector{Vector{Bool}}(l)
	t = time()
	if elType <: Nullable
		for i in 1:l
			out = predictNullTree(graph, input, i)
			outputs[i] = out[1]
			exampleCounts[i] = out[2]
			paths[i] = out[3]
		end
	else	
		for i in 1:l
			out = predictNonNullTree(graph, input, i, Vector{Bool}(), Vector{Float64}(), Vector{Int64}())
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
	end

	(outputs, exampleCounts, paths)
end

function runSimplePredictTree(graph::DecisionTree, input::Vector{T}) where T <: Any
	featureType = typeof(input[1])
	assert(featureType <: AbstractVector)
	elType = typeof(input[1][1])
	assert(isbits(elType))
	m = length(input)
	l = length(input[1])
	if m > 1
		for i = 2:m
			assert(length(input[i]) == l)
		end
	end

	output = Vector{Float64}(l)
	t = time()
	if elType <: Nullable
		for i in 1:l
			output[i] = predictSimpleNullTree(graph, input, i)
		end
	else	
		for i in 1:l
			output[i] = predictSimpleNonNullTree(graph, input, i)
			if time() - t > 5
				t = time()
				print("\r\u1b[K\u1b[A")
				print("\r")
				print("\u1b[K")
				println(string("Done with input ", i, " of ", l))
			end
		end
	end

	output
end

function predictNonNullTree(graph::DecisionTree, input::Vector{T}, i::Int64, path::Vector{Bool}, predictions::Vector{Float64}, counts::Vector{Int64}) where T <: Any
	if typeof(graph) <: DecisionLeaf
		([predictions; Float64(graph.prediction)], [counts; length(graph.ind)], path)
	elseif typeof(graph) <: DecisionLabelNode
		in(input[graph.splitCol][i], graph.leftPartition) ? predictNonNullTree(graph.left, input, i, [path; false], [predictions; Float64(graph.leaf.prediction)], [counts; length(graph.leaf.ind)]) : predictNonNullTree(graph.right, input, i, [path; true], [predictions; Float64(graph.leaf.prediction)], [counts; length(graph.leaf.ind)])
	else
		input[graph.splitCol][i] <= graph.splitPoint ? predictNonNullTree(graph.left, input, i, [path; false], [predictions; Float64(graph.leaf.prediction)], [counts; length(graph.leaf.ind)]) : predictNonNullTree(graph.right, input, i, [path; true], [predictions; Float64(graph.leaf.prediction)], [counts; length(graph.leaf.ind)])
	end
end

function predictSimpleNonNullTree(graph::DecisionTree, input::Vector{T}, i::Int64) where T <: Any
	if typeof(graph) <: Union{DecisionLeaf, DecisionTestTrainLeaf}
		Float64(graph.prediction)
	elseif typeof(graph) <: Union{DecisionLabelNode, DecisionTestTrainLabelNode}
		in(input[graph.splitCol][i], graph.leftPartition) ? predictSimpleNonNullTree(graph.left, input, i) : predictSimpleNonNullTree(graph.right, input, i)
	else
		input[graph.splitCol][i] <= graph.splitPoint ? predictSimpleNonNullTree(graph.left, input, i) : predictSimpleNonNullTree(graph.right, input, i)
	end
end

function predictNullTree(graph::DecisionTree, input::Vector{T}, i::Int64, r = 1) where T <: Any
	if typeof(graph) <: DecisionLeaf
		(graph.prediction, length(graph.ind), r)
	elseif typeof(graph) <: DecisionNullLabelNode
		if input[graph.splitCol][i].hasvalue
			assert(typeof(input[graph.splitCol][i].value) == Int64)
			in(input[graph.splitCol][i].value, graph.leftPartition) ? predictNullTree(graph.left, input, i, r+1) : predictNullTree(graph.right, input, i, r+1)
		else
			graph.nullLeft ? predictNullTree(graph.left, input, i, r+1) : predictNullTree(graph.right, input, i, r+1)
		end
	elseif typeof(graph) <: DecisionNullNode
		if input[graph.splitCol][i].hasvalue
			input[graph.splitCol][i].value <= graph.splitPoint ? predictNullTree(graph.left, input, i, r+1) : predictNullTree(graph.right, input, i, r+1)
		else
			graph.nullLeft ? predictNullTree(graph.left, input, i, r+1) : predictNullTree(graph.right, input, i, r+1)
		end
	else
		error("input contains null values which tree is not designed to handle")
	end
end

function predictTree(graph::DecisionTree, input::T, r = 1)  where T <: DenseArray{U, 1} where U <: Union{Real, Missing}
	if typeof(graph) <: DecisionLeaf
		(graph.prediction, length(graph.ind), r)
	elseif typeof(graph) <: DecisionNullLabelNode
		if input[graph.splitCol].hasvalue
			assert(typeof(input[graph.splitCol].value) == Int64)
			in(input[graph.splitCol].value, graph.leftPartition) ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		else
			graph.nullLeft ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		end
	elseif typeof(graph) <: DecisionNullNode
		if input[graph.splitCol].hasvalue
			input[graph.splitCol].value <= graph.splitPoint ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		else
			graph.nullLeft ? predictTree(graph.left, input, r+1) : predictTree(graph.right, input, r+1)
		end
	else
		error("input contains null values which tree is not designed to handle")
	end
end




function predictForest(forest::decisionForest, input::Vector{T}) where T <: Any
	out = [runSimplePredictTree(a[2], input) for a in forest]
	meanOut = reduce((a, b) -> a.+b, out)/length(out)
	stdOut = sqrt.(mapreduce(a -> (a.-meanOut).^2, (a, b) -> a.+b, out)/length(out))
	(meanOut, stdOut)
end


#--------------------------------Conversion Functions-----------------------------------------
function convTreeGraph(tree::DecisionTree{T}) where T <: Real
	if typeof(tree) == DecisionLeaf{T}
		DecisionGraphTerminus(tree.prediction)
	else
		DecisionGraphNode(tree.leaf.prediction, tree.splitCol, tree.splitPoint, convTreeGraph(tree.left), convTreeGraph(tree.right))
	end
end

function convLabelTreeGraph(tree::DecisionTree{T}) where T <: Real
	if typeof(tree) == DecisionLeaf{T}
		DecisionGraphTerminus(tree.prediction)
	else
		DecisionGraphLabelNode(tree.leaf.prediction, tree.splitCol, tree.leftPartition, convLabelTreeGraph(tree.left), convLabelTreeGraph(tree.right))
	end
end

