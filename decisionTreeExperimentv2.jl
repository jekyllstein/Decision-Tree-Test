using Random
using Statistics

include("decisionTreeUtils.jl")
include("inputSplitFunctions.jl")
#-----------------------------------------------Single Tree Formation-------------------------------------------------------
function makeRegTree(input::Vector{T}, output::Vector{T}, sqOutput::Vector{T}, selectedInd::Vector{Int64}; p = 5, q = 1) where T <: Real 
	S = var(output[selectedInd])*length(selectedInd)
	leaf = DecisionLeaf(selectedInd, mean(output[selectedInd]), S/length(selectedInd))
	if length(selectedInd) <= p
		leaf
	else
		(splitPoint, ind1, ind2, minValue) = findRegSplit(input, output, sqOutput, selectedInd)
		if (S - minValue) < q
			leaf
		else
			DecisionNode(leaf, 1, splitPoint, makeRegTree(input, output, sqOutput, ind1, p = p, q = q), makeRegTree(input, output, sqOutput, ind2, p = p, q = q))
		end
	end
end

function makeBoolTree(input::Vector{T}, output::U, selectedInd::V = eachindex(output); p = 10) where T <: Real where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		(splitPoint, ind1, ind2, minValue) = findBoolSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			DecisionNode(leaf, 1, splitPoint, makeBoolTree(input, output, ind1, p = p), makeBoolTree(input, output, ind2, p = p))
		end
	end
end

function makeBoolLabelTree(input::Vector{Int64}, output::U, selectedInd::V = eachindex(output); p = 10) where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		(leftPartition, ind1, ind2, minValue) = findBoolSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			DecisionLabelNode(leaf, 1, leftPartition, makeBoolLabelTree(input, output, ind1, p = p), makeBoolLabelTree(input, output, ind2, p = p))
		end
	end
end

function makeBoolLabelTree(input::Vector{Vector{Int64}}, output::U, selectedInd::V = eachindex(output); p = 10) where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		splitOut = [findBoolSplit(v, output, selectedInd) for v in input]
		minValues = [a[4] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(leftPartition, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			DecisionLabelNode(leaf, minInd, leftPartition, makeBoolLabelTree(input, output, ind1, p = p), makeBoolLabelTree(input, output, ind2, p = p))
		end
	end
end

function makeDecisionTree(input::vectorInputType, output::outputBoolType; inputSortInds::Vector{Vector{Int64}} = [sortperm(v) for v in input], inputInvSortInds::Vector{Vector{Int64}} = [sortperm(v) for v in inputSortInds], selectedInd::AbstractVector{Bool} = fill(true, length(output)), p = 10)
	orderedInd = findall(selectedInd)
	E = calcBoolEntropy(output[orderedInd])
	pred = mean(output[orderedInd])
	leaf = DecisionLeaf(orderedInd, pred, typeof(pred)(E))
	if (length(orderedInd) <= p) | (E == 0)
		leaf
	else
		splitOut = [findBoolSplit(input[i], output, inputSortInds[i], inputInvSortInds[i], selectedInd) for i in 1:length(input)]
		minValues = [a[4] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(splitCond, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (count(ind1) < p) | (count(ind2) < p) | (minValue == E)
			leaf
		else
			nodeType = (typeof(splitCond) == Set{Int64}) ? DecisionLabelNode : DecisionNode
			nodeType(leaf, minInd, splitCond, map(ind -> makeDecisionTree(input, output, inputSortInds = inputSortInds, inputInvSortInds = inputInvSortInds, selectedInd = ind, p = p), (ind1, ind2))...) 
		end
	end
end

function runMakeDecisionTree(input::vectorInputType, output::outputNumType, sqOutput::outputNumType; p = 10, t = 1e-4)
	inputSortInds = [sortperm(v) for v in input]
	inputInvSortInds = [sortperm(v) for v in inputSortInds]
	selectedInd = fill(true, length(output))
	# println("hi")
	makeDecisionTree(input, output, sqOutput, inputSortInds, inputInvSortInds, selectedInd, p, t)
end

function makeDecisionTree(input::vectorInputType, output::outputNumType, sqOutput::outputNumType, inputSortInds::Vector{Vector{Int64}}, inputInvSortInds::Vector{Vector{Int64}}, selectedInd::AbstractVector{Bool}, p, t)
	numInd = count(selectedInd)
	# orderedInd = findall(selectedInd)
	# pred = mean(view(output, orderedInd))
	pred = selectSum(output, selectedInd)/numInd
	# E = (mean(view(sqOutput, orderedInd)) - pred^2)*length(orderedInd)
	E = (selectSum(sqOutput, selectedInd)/numInd - pred^2)*numInd
	# leaf = DecisionLeaf(orderedInd, pred, typeof(pred)(E))
	leaf = DecisionLeaf(ones(Int64, numInd), pred, typeof(pred)(E))
	if (numInd <= p) | (E == 0)
		leaf
	else
		splitOut = [findRegSplit(Vector{typeof(input[i][1])}(input[i]), output, sqOutput, inputSortInds[i], inputInvSortInds[i], selectedInd) for i in 1:length(input)]
		minValues = [a[4] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(splitCond, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (count(ind1) < p) | (count(ind2) < p) | ((E - minValue)/E < t)
			leaf
		else
			if typeof(splitCond) == Set{Int64}
				DecisionLabelNode(leaf, minInd, splitCond, makeDecisionTree(input, output, sqOutput, inputSortInds, inputInvSortInds, ind1, p, t), makeDecisionTree(input, output, sqOutput, inputSortInds, inputInvSortInds, ind2, p, t))
			else
				DecisionNode(leaf, minInd, splitCond, makeDecisionTree(input, output, sqOutput, inputSortInds, inputInvSortInds, ind1, p, t), makeDecisionTree(input, output, sqOutput, inputSortInds, inputInvSortInds, ind2, p, t))
			end
		end
	end
end

function makeDecisionTree(input::T, output::outputBoolType, selectedInd::V = eachindex(output); p = 10) where T <: vectorInputType where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		splitOut = [findBoolSplit(v, output, selectedInd) for v in input]
		minValues = [a[4] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(splitCond, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			if typeof(splitCond) == Set{Int64}
				DecisionLabelNode(leaf, minInd, splitCond, makeDecisionTree(input, output, ind1, p = p), makeDecisionTree(input, output, ind2, p = p))
			else
				DecisionNode(leaf, minInd, splitCond, makeDecisionTree(input, output, ind1, p = p), makeDecisionTree(input, output, ind2, p = p))
			end
		end
	end
end

function makeDecisionTree(input::Vector{T}, output::U, sqOutput::U, selectedInd::V = eachindex(output); p = 10, t = 1e-4, sortedInd = [eltype(v) <: AbstractFloat ? sortperm(v) : Vector{Int64}() for v in input]) where T <: Any where U <: outputNumType where V <: AbstractArray{Int64, 1}
	pred = mean(view(output, selectedInd))
	E = (mean(view(sqOutput, selectedInd)) - pred^2)*length(selectedInd)
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E/length(selectedInd)))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		# splitOut = [typeof(input[i][1]) <: AbstractFloat ? findRegSplit(Vector{typeof(input[i][1])}(input[i]), output, sqOutput, selectedInd, sortedInd[i]) : findRegSplit(Vector{typeof(input[i][1])}(input[i]), output, sqOutput, selectedInd) for i in 1:length(input)]
		splitOut = [typeof(input[i][1]) <: AbstractFloat ? findRegSplit(Vector{typeof(input[i][1])}(input[i]), output, sqOutput, selectedInd, sortedInd[i]) : findRegSplit(Vector{typeof(input[i][1])}(input[i]), output, sqOutput, selectedInd) for i in 1:length(input)]
		minValues = [a[4] for a in splitOut]
		minSplitSize = [min(length(a[2]), length(a[3])) for a in splitOut]
		validInd = findall(minSplitSize .>= p)
		if isempty(validInd)
			println("No splits of sufficient size, creating leaf")
			leaf
		else
			(minValue, minValidInd) = findmin(minValues[validInd])
			minInd = validInd[minValidInd]
			(splitCond, ind1, ind2, minValue) = splitOut[minInd]
			println(string("Split Ind : ", minInd, " with left numInd = ", length(ind1), " and right num Ind = ", length(ind2))) 
			# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
			if ((E - minValue)/E < t)
				println("Best split of valid size has error reduction less than threshold, creating leaf")
				leaf
			else
				if typeof(splitCond) == Set{Int64}
					DecisionLabelNode(leaf, minInd, splitCond, makeDecisionTree(input, output, sqOutput, ind1, p = p, t = t, sortedInd = sortedInd), makeDecisionTree(input, output, sqOutput, ind2, p = p, t = t, sortedInd = sortedInd))
				else
					DecisionNode(leaf, minInd, splitCond, makeDecisionTree(input, output, sqOutput, ind1, p = p, t = t, sortedInd = sortedInd), makeDecisionTree(input, output, sqOutput, ind2, p = p, t = t, sortedInd = sortedInd))
				end
			end
		end
	end
end

function makeTestDecisionTree(input::Vector{T}, output::AbstractArray{U}, sqOutput::AbstractArray{U}, testInput::Vector{T}, testOutput::AbstractArray{U}, sqTestOutput::AbstractArray{U}; selectedInd::V = eachindex(output), testSelectedInd::V = eachindex(testOutput), p = 10, t = 1e-4, sortedInd = [eltype(v) <: AbstractFloat ? sortperm(v) : Vector{Int64}() for v in input]) where T <: Any where U <: Real where V <: AbstractArray{Int64, 1}
	pred = mean(view(output, selectedInd))
	E = (mean(view(sqOutput, selectedInd)) -  pred^2)*length(selectedInd)
	testE = mean((view(testOutput, testSelectedInd) -  pred).^2)

	leaf = DecisionTestTrainLeaf(collect(selectedInd), pred, typeof(pred)(E)/length(selectedInd), collect(testSelectedInd), typeof(pred)(testE))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		splitOut = map(i -> typeof(input[i][1]) <: AbstractFloat ? findRegSplit(input[i], output, sqOutput, selectedInd, sortedInd[i]) : findRegSplit(input[i], output, sqOutput, selectedInd), 1:length(input))
		# println("hi")
		# splitOut = [findRegSplit(Vector{typeof(v[1])}(v), output, sqOutput, selectedInd) for v in view(input, usedFeatures)]
		minValues = [a[4] for a in splitOut]
		minSplitSize = [min(length(a[2]), length(a[3])) for a in splitOut]
		validInd = findall(minSplitSize .>= p)
		if isempty(validInd)
			println("No splits of sufficient size, creating leaf")
			leaf
		else
			(minValue, minValidInd) = findmin(minValues[validInd])
			minInd = validInd[minValidInd]
			(splitCond, ind1, ind2, minValue) = splitOut[minInd]
			# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
			println(string("Split Ind : ", minInd, " with left numInd = ", length(ind1), " and right num Ind = ", length(ind2))) 
			if (E - minValue)/E < t
				println("Best split of valid size has error reduction less than threshold, creating leaf")
				leaf
			else
				if typeof(splitCond) == Set{Int64}
					(testInd1, testInd2) = if isempty(testSelectedInd)
						(Vector{Int64}(), Vector{Int64}())
					else
						testSplits = [in(a, splitCond) for a in view(testInput[minInd], testSelectedInd)]
						testInd1 = testSelectedInd[findall(testSplits)]
						testInd2 = testSelectedInd[findall(a -> !a, testSplits)]
						(testInd1, testInd2)
					end
					DecisionTestTrainLabelNode(leaf, minInd, splitCond, makeTestDecisionTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd = ind1, testSelectedInd = testInd1, p = p, t = t, sortedInd = sortedInd), makeTestDecisionTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd=ind2, testSelectedInd=testInd2, p = p, t = t, sortedInd = sortedInd))
				else
					(testInd1, testInd2) = if isempty(testSelectedInd)
						(Vector{Int64}(), Vector{Int64}())
					else
						testSplits = [a <= splitCond for a in view(testInput[minInd], testSelectedInd)]
						testInd1 = testSelectedInd[findall(testSplits)]
						testInd2 = testSelectedInd[findall(a -> !a, testSplits)]
						(testInd1, testInd2)
					end
					leftTree = makeTestDecisionTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd = ind1, testSelectedInd = testInd1, p = p, t = t, sortedInd = sortedInd)
					rightTree = makeTestDecisionTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd=ind2, testSelectedInd=testInd2, p = p, t = t, sortedInd = sortedInd)
					DecisionTestTrainNode(leaf, minInd, splitCond, leftTree, rightTree)
				end
			end
		end
	end
end

function makeDecisionTree(input::T, output::U, selectedInd::V = eachindex(output); p = 10) where T <: vectorInputType where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		splitOut = [findBoolSplit(v, output, selectedInd) for v in input]
		minValues = [a[5] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(splitCond, nullLeft, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			if typeof(splitCond) == Set{Int64}
				DecisionNullLabelNode(leaf, minInd, splitCond, nullLeft, makeDecisionTree(input, output, ind1, p = p), makeDecisionTree(input, output, ind2, p = p))
			else
				DecisionNullNode(leaf, minInd, splitCond, nullLeft, makeDecisionTree(input, output, ind1, p = p), makeDecisionTree(input, output, ind2, p = p))
			end
		end
	end
end

function makeRandomSubspaceTree(input::T, output::U, selectedInd::V = eachindex(output); p = 10) where T <: vectorInputType where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	f = length(input) #number of feature vectors
	n = floor(Int64, sqrt(f)) #number of candidate features at each split
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		usedFeatures = shuffle(1:f)[1:n]
		splitOut = [findBoolSplit(v, output, selectedInd) for v in view(input, usedFeatures)]
		minValues = [a[4] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(splitCond, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			if typeof(splitCond) == Set{Int64}
				DecisionLabelNode(leaf, usedFeatures[minInd], splitCond, makeRandomSubspaceTree(input, output, ind1, p = p), makeRandomSubspaceTree(input, output, ind2, p = p))
			else
				DecisionNode(leaf, usedFeatures[minInd], splitCond, makeRandomSubspaceTree(input, output, ind1, p = p), makeRandomSubspaceTree(input, output, ind2, p = p))
			end
		end
	end
end

function makeRandomSubspaceTree(input::Vector{T}, output::AbstractArray{U}, sqOutput::AbstractArray{U}; selectedInd::V = eachindex(output), p = 10, t = 1e-4, sortedInd = [eltype(v) <: AbstractFloat ? sortperm(v) : Vector{Int64}() for v in input]) where T <: Any where U <: Real where V <: AbstractArray{Int64, 1}
	f = length(input) #number of feature vectors
	n = max(min(f, 2), floor(Int64, sqrt(f))) #number of candidate features at each split
	pred = mean(view(output, selectedInd))
	E = (mean(view(sqOutput, selectedInd)) -  pred^2)*length(selectedInd)
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E)/length(selectedInd))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		usedFeatures = shuffle(1:f)[1:n]
		usedInput = [input[i] for i in usedFeatures]
		splitOut = map(i -> typeof(input[i][1]) <: AbstractFloat ? findRegSplit(input[i], output, sqOutput, selectedInd, sortedInd[i]) : findRegSplit(input[i], output, sqOutput, selectedInd), usedFeatures)
		
		# splitOut = [findRegSplit(Vector{typeof(v[1])}(v), output, sqOutput, selectedInd) for v in view(input, usedFeatures)]
		minValues = [a[4] for a in splitOut]
		minSplitSize = [min(length(a[2]), length(a[3])) for a in splitOut]
		validInd = findall(minSplitSize .>= p)
		if isempty(validInd)
			# println("No splits of sufficient size, creating leaf")
			leaf
		else
			(minValue, minValidInd) = findmin(minValues[validInd])
			minInd = validInd[minValidInd]
			(splitCond, ind1, ind2, minValue) = splitOut[minInd]
			# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
			# println(string("Split Ind : ", usedFeatures[minInd], " with left numInd = ", length(ind1), " and right num Ind = ", length(ind2))) 
			if (E - minValue)/E < t
				# println("Best split of valid size has error reduction less than threshold, creating leaf")
				leaf
			else
				if typeof(splitCond) == Set{Int64}
					DecisionLabelNode(leaf, usedFeatures[minInd], splitCond, makeRandomSubspaceTree(input, output, sqOutput, p = p, t = t, selectedInd = ind1, sortedInd = sortedInd), makeRandomSubspaceTree(input, output, sqOutput, selectedInd = ind2, p = p, t = t, sortedInd = sortedInd))
				else
					DecisionNode(leaf, usedFeatures[minInd], splitCond, makeRandomSubspaceTree(input, output, sqOutput, selectedInd = ind1, p = p, t = t, sortedInd = sortedInd), makeRandomSubspaceTree(input, output, sqOutput, selectedInd = ind2, p = p, t = t, sortedInd = sortedInd))
				end
			end
		end
	end
end

function makeRandomSubspaceTree(input::Vector{T}, output::AbstractArray{U}, sqOutput::AbstractArray{U},  testInput::Vector{T}, testOutput::AbstractArray{U}, sqTestOutput::AbstractArray{U}; selectedInd::V = eachindex(output), testSelectedInd::W = eachindex(testOutput), p = 10, t = 1e-4, sortedInd = [eltype(v) <: AbstractFloat ? sortperm(v) : Vector{Int64}() for v in input]) where T <: Any where U <: Real where V <: AbstractArray{Int64, 1} where W <: AbstractArray{Int64, 1}
	
	f = length(input) #number of feature vectors
	n = max(min(f, 2), floor(Int64, sqrt(f))) #number of candidate features at each split
	pred = mean(view(output, selectedInd))
	E = (mean(view(sqOutput, selectedInd)) -  pred^2)*length(selectedInd)
	testE = mean((view(testOutput, testSelectedInd) -  pred).^2)

	leaf = DecisionTestTrainLeaf(collect(selectedInd), pred, typeof(pred)(E)/length(selectedInd), collect(testSelectedInd), typeof(pred)(testE))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		usedFeatures = shuffle(1:f)[1:n]
		usedInput = [input[i] for i in usedFeatures]
		splitOut = [typeof(input[i][1]) <: AbstractFloat ? findRegSplit(input[i], output, sqOutput, selectedInd, sortedInd[i]) : findRegSplit(input[i], output, sqOutput, selectedInd) for i in usedFeatures]
		
		# splitOut = [findRegSplit(Vector{typeof(v[1])}(v), output, sqOutput, selectedInd) for v in view(input, usedFeatures)]
		minValues = [a[4] for a in splitOut]
		minSplitSize = [min(length(a[2]), length(a[3])) for a in splitOut]
		validInd = findall(minSplitSize .>= p)
		if isempty(validInd)
			# println("No splits of sufficient size, creating leaf")
			leaf
		else
			(minValue, minValidInd) = findmin(minValues[validInd])
			minInd = validInd[minValidInd]
			(splitCond, ind1, ind2, minValue) = splitOut[minInd]
			# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
			# println(string("Split Ind : ", usedFeatures[minInd], " with left numInd = ", length(ind1), " and right num Ind = ", length(ind2))) 
			if (E - minValue)/E < t
				# println("Best split of valid size has error reduction less than threshold, creating leaf")
				leaf
			else
				if typeof(splitCond) == Set{Int64}
					(testInd1, testInd2) = if isempty(testSelectedInd)
						(Vector{Int64}(), Vector{Int64}())
					else
						testSplits = [in(a, splitCond) for a in view(testInput[usedFeatures[minInd]], testSelectedInd)]
						testInd1 = testSelectedInd[findall(testSplits)]
						testInd2 = testSelectedInd[findall(a -> !a, testSplits)]
						(testInd1, testInd2)
					end
					DecisionTestTrainLabelNode(leaf, usedFeatures[minInd], splitCond, makeRandomSubspaceTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd = ind1, testSelectedInd = testInd1, p = p, t = t, sortedInd = sortedInd), makeRandomSubspaceTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd=ind2, testSelectedInd=testInd2, p = p, t = t, sortedInd = sortedInd))
				else
					(testInd1, testInd2) = if isempty(testSelectedInd)
						(Vector{Int64}(), Vector{Int64}())
					else
						testSplits = [a <= splitCond for a in view(testInput[usedFeatures[minInd]], testSelectedInd)]
						testInd1 = testSelectedInd[findall(testSplits)]
						testInd2 = testSelectedInd[findall(a -> !a, testSplits)]
						(testInd1, testInd2)
					end
					DecisionTestTrainNode(leaf, usedFeatures[minInd], splitCond, makeRandomSubspaceTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd = ind1, testSelectedInd = testInd1, p = p, t = t, sortedInd = sortedInd), makeRandomSubspaceTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, selectedInd=ind2, testSelectedInd=testInd2, p = p, t = t, sortedInd = sortedInd))
				end
			end
		end
	end
end


function makeRandomSubspaceTree(input::T, output::U, selectedInd::V = eachindex(output); p = 10) where T <: vectorInputType where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	f = length(input) #number of feature vectors
	n = floor(Int64, sqrt(f)) #number of candidate features at each split
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		usedFeatures = shuffle(1:f)[1:n]
		splitOut = [findBoolSplit(v, output, selectedInd) for v in view(input, usedFeatures)]
		minValues = [a[5] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(splitCond, nullLeft, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			if typeof(splitCond) == Set{Int64}
				DecisionNullLabelNode(leaf, usedFeatures[minInd], splitCond, nullLeft, makeRandomSubspaceTree(input, output, ind1, p = p), makeRandomSubspaceTree(input, output, ind2, p = p))
			else
				DecisionNullNode(leaf, usedFeatures[minInd], splitCond, nullLeft, makeRandomSubspaceTree(input, output, ind1, p = p), makeRandomSubspaceTree(input, output, ind2, p = p))
			end
		end
	end
end

function makeExtraTree(input::T, output::U, selectedInd::V = eachindex(output); p = 10) where T <: inputType2 where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	f = length(input) #number of feature vectors
	n = floor(Int64, sqrt(f)) #number of candidate features at each split
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		usedFeatures = shuffle(1:f)[1:n]
		splitOut = [randomBoolSplit(v, output, selectedInd) for v in view(input, usedFeatures)]
		minValues = [a[4] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(splitCond, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			if typeof(splitCond) == Set{Int64}
				DecisionLabelNode(leaf, usedFeatures[minInd], splitCond, makeExtraTree(input, output, ind1, p = p), makeExtraTree(input, output, ind2, p = p))
			else
				DecisionNode(leaf, usedFeatures[minInd], splitCond, makeExtraTree(input, output, ind1, p = p), makeExtraTree(input, output, ind2, p = p))
			end
		end
	end
end

function makeBoolLabelRandomSubspaceTree(input::Vector{Vector{Int64}}, output::U, selectedInd::V = eachindex(output); p = 10) where U <: DenseArray{Bool, 1} where V <: AbstractArray{Int64, 1}
	f = length(input) #number of feature vectors
	n = floor(Int64, sqrt(f)) #number of candidate features at each split
	E = calcBoolEntropy(output[selectedInd])
	pred = mean(output[selectedInd])
	leaf = DecisionLeaf(collect(selectedInd), pred, typeof(pred)(E))
	if (length(selectedInd) <= p) | (E == 0)
		leaf
	else
		usedFeatures = shuffle(1:f)[1:n]
		splitOut = [findBoolSplit(v, output, selectedInd) for v in view(input, usedFeatures)]
		minValues = [a[4] for a in splitOut]
		(minValue, minInd) = findmin(minValues)
		(leftPartition, ind1, ind2, minValue) = splitOut[minInd]
		# (leftPartition, ind1, ind2, minValue) = findBoolLabelSplit(input, output, selectedInd)
		if (length(ind1) < p) | (length(ind2) < p) | (minValue == E)
			leaf
		else
			DecisionLabelNode(leaf, usedFeatures[minInd], leftPartition, makeBoolLabelRandomSubspaceTree(input, output, ind1, p = p), makeBoolLabelRandomSubspaceTree(input, output, ind2, p = p))
		end
	end
end

function makeBoolLabelRandomForest(input::Vector{Vector{Int64}}, output::U; p = 10, n = length(output), B = 100, seed = 1234) where U <: DenseArray{Bool, 1}
	l = length(output)
	Random.seed!(seed)
	[begin
		bootstrapInds = ceil.(Int64, rand(n)*l)
	    inputSample = [v[bootstrapInds] for v in input]
		outputSample = output[bootstrapInds]
		println(string("Creating tree for bootstrap sample ", i))
		(bootstrapInds, makeBoolLabelRandomSubspaceTree(inputSample, outputSample, p = p))
	end
	for i in 1:B]
end

#----------------------------------Tree pruning-------------------------------------
function mapTerminalBranches(tree::DecisionTree; pathList=Vector{Vector{Bool}}(), path=Vector{Bool}())
#iterate through tree and generate a list of "paths" to nodes with two leafs as children.  The paths are a boolean
#vector where false indicates a branch left and true a branch right
	if typeof(tree.left) <: DecisionTerm
		if typeof(tree.right) <: DecisionTerm
			[pathList; [path]]
		else
			mapTerminalBranches(tree.right, pathList=pathList, path = [path; true])
		end
	else
		leftPathList = mapTerminalBranches(tree.left, pathList=pathList, path = [path; false])
		if typeof(tree.right) <: DecisionTerm
			leftPathList
		else
			mapTerminalBranches(tree.right, pathList=leftPathList, path = [path; true])
		end
	end
end

#If tree is already a leaf then leave unchanged
function pruneTree(tree::DecisionTerm, path::Vector{Bool}, pathCount=1)
	tree
end

function pruneTree(tree::DecisionBranch, path::Vector{Bool}, pathCount=1)
#generate a pruned tree by condensing the branch reached by following "path" into a leaf
	if pathCount == length(path)
		DecisionLeaf(tree.leaf.ind, tree.leaf.prediction, tree.leaf.err)
	else
		#keep all fields of current node except the last 2 which are the left and right branch
		baseInputs = map(i -> getfield(tree, i), (1, 2, 3, 4))
		if path[pathCount]
			typeof(tree)(baseInputs..., tree.left, pruneTree(tree.right, path, pathCount+1)) 
		else
			typeof(tree)(baseInputs..., pruneTree(tree.left, path, pathCount+1), tree.right)
		end
	end
end 

function pruneDecisionTree(tree::DecisionTree, testInput::vectorInputType, testOutput::outputBoolType, previousErr = calcClassErr([predictTree(tree, v)[1] for v in testInput] .> 0.5, testOutput))
	if typeof(tree) <: DecisionLeaf
		(tree, previousErr)
	else
		pathList = filter(!isempty, mapTerminalBranches(tree))
		if !isempty(pathList)
			prunedResults = [begin
			    prunedTree = pruneTree(tree, path)
			    testTreeOutput = [predictTree(prunedTree, v)[1] for v in testInput]
			    (calcClassErr(testTreeOutput .> 0.5, testOutput), prunedTree)
			end
			for path in pathList]

			testErrs = [a[1] for a in prunedResults]
			
			(minValue, minInd) = findmin(testErrs)
			if minValue > previousErr
				(tree, previousErr)
			else
				if (typeof(prunedResults[minInd][2]) <: DecisionLeaf)
					(prunedResults[minInd][2], minValue)
				else
					pruneDecisionTree(prunedResults[minInd][2], testInput, testOutput, minValue)
				end
			end
		else
			(tree, previousErr)
		end
	end
end

function pruneDecisionTree(tree::DecisionTree, testInput::Vector{T}, testOutput::U, previousErr = calcOutputErr(runSimplePredictTree(tree, testInput), testOutput)) where T <: Any where U <: outputNumType
	pathList = filter(!isempty, mapTerminalBranches(tree))
	println(string("Number of paths to test:", length(pathList)))
	function checkPaths(rootTree, previousErr, pathList, i=1)
		if isempty(pathList)
			return (previousErr, rootTree, i)
		else
			prunedTree = pruneTree(rootTree, pathList[1])
			testTreeOutput = runPredictTree(prunedTree, testInput)[1]
			testErr = calcOutputErr(testTreeOutput, testOutput)
			if testErr <= previousErr
				if length(pathList) == 1
					(testErr, prunedTree, i+1)
				else
					checkPaths(prunedTree, testErr, pathList[2:end], i+1)
				end
			else
				if length(pathList) == 1
					(previousErr, rootTree, i)
				else
					checkPaths(rootTree, previousErr, pathList[2:end], i)
				end
			end
		end
	end
	
	out = checkPaths(tree, previousErr, pathList)

	# testErrs = [a[1] for a in prunedResults]
	# (minValue, minInd) = findmin(testErrs)
	println(out[3] - 1, " branches pruned out of ", length(pathList), " attempts")
	if out[3] == 1
		(tree, previousErr)
	else
		if (typeof(out[2]) <: DecisionLeaf)
			(out[2], out[1])
		else
			pruneDecisionTree(out[2], testInput, testOutput, out[1])
		end
	end
end

function pruneTestTrainDecisionTree(tree::DecisionTestTrainTree, previousErr = calcTreeTestError(tree))
	pathList = filter(!isempty, mapTerminalBranches(tree))
	println(string("Number of paths to test:", length(pathList)))
	function checkPaths(rootTree, previousErr, pathList, i=1)
		if isempty(pathList)
			return (previousErr, rootTree, i)
		else
			prunedTree = pruneTree(rootTree, pathList[1])
			testErr = calcTreeTestError(prunedTree)
			if testErr <= previousErr
				if length(pathList) == 1
					(testErr, prunedTree, i+1)
				else
					checkPaths(prunedTree, testErr, pathList[2:end], i+1)
				end
			else
				if length(pathList) == 1
					(previousErr, rootTree, i)
				else
					checkPaths(rootTree, previousErr, pathList[2:end], i)
				end
			end
		end
	end
	
	out = checkPaths(tree, previousErr, pathList)

	# testErrs = [a[1] for a in prunedResults]
	# (minValue, minInd) = findmin(testErrs)
	println(out[3] - 1, " branches pruned out of ", length(pathList), " attempts")
	if out[3] == 1
		(tree, previousErr, formOrderedTestTrainTreeOutput(tree)[2])
	else
		if (typeof(out[2]) <: DecisionTestTrainLeaf)
			(out[2], out[1], formOrderedTestTrainTreeOutput(out[2])[2])
		else
			pruneTestTrainDecisionTree(out[2], out[1])
		end
	end
end

function pruneDecisionTree(tree::DecisionTree, testInput::Vector{T}, testOutput::U; testPrediction::Tuple = typeof(tree) <: DecisionLeaf ? () : runPredictTree(tree, testInput), testPredictionOutput::Vector{Float64} = isempty(testPrediction) ? runSimplePredictTree(tree, testInput) : [a[end] for a in testPrediction[1]], previousErr::Float64 = calcOutputErr(testPredictionOutput, testOutput), pastPaths=Vector{Vector{Bool}}()) where T <: Any where U <: outputNumType
	pathList = setdiff(filter(!isempty, mapTerminalBranches(tree)), pastPaths)
	println(string("Previous test error = ", previousErr))
	println(string("Number of paths to test:", length(pathList)))
	testOutputs = testPrediction[1]
	testPaths = testPrediction[3]
	

	function calcPrunedOutput(testOutput, testPath, prunePath)
		l1 = length(prunePath)
		l2 = length(testPath)
		if l2 > l1
			testPath[1:l1] == prunePath ? testOutput[l1+1] : testOutput[l2+1]
		else
			testOutput[l2+1]
		end
	end


	function checkPaths(rootTree, previousTestTreeOutput, previousErr, pathList, i=1; t = time())
		if isempty(pathList)
			return (previousErr, rootTree, previousTestTreeOutput, i)
		else
			if time() - t > 5
				print("\r\u1b[K\u1b[A")
				print("\r")
				print("\u1b[K")
				println(string("Number of paths to test:", length(pathList)))
				t = time()
			end
			path = pathList[1]
			prunedTree = pruneTree(rootTree, path)
			prunedTestTreeOutput = [calcPrunedOutput(testOutputs[j], testPaths[j], path) for j in 1:length(testOutputs)]
			testErr = calcOutputErr(prunedTestTreeOutput, testOutput)
			# println(string("Pruned Test Err = ", testErr))
			if testErr <= previousErr
				if length(pathList) == 1
					(testErr, prunedTree, prunedTestTreeOutput, i+1)
				else
					checkPaths(prunedTree, prunedTestTreeOutput, testErr, pathList[2:end], i+1, t = t)
				end
			else
				if length(pathList) == 1
					(previousErr, rootTree, previousTestTreeOutput, i)
				else
					checkPaths(rootTree, previousTestTreeOutput, previousErr, pathList[2:end], i, t = t)
				end
			end
		end
	end
	
	out = checkPaths(tree, testPredictionOutput, previousErr, pathList)

	# testErrs = [a[1] for a in prunedResults]
	# (minValue, minInd) = findmin(testErrs)
	println(out[4] - 1, " branches pruned out of ", length(pathList), " attempts")
	if out[4] == 1
		(tree, previousErr, testPredictionOutput)
	else
		if (typeof(out[2]) <: DecisionLeaf)
			(out[2], out[1], out[3])
		else
			pruneDecisionTree(out[2], testInput, testOutput, testPrediction = testPrediction, testPredictionOutput = out[3], previousErr = out[1], pastPaths = [pastPaths; pathList])
		end
	end
end


#---------------------------------------------Random Forest-----------------------------------------------

function makeRandomForest(input::vectorInputType, output::outputBoolType; p = 10, n = length(output), B = 100, seed = 1234)
	l = length(output)
	Random.seed!(seed)
	[begin
		bootstrapInds = ceil.(Int64, rand(n)*l)
	    inputSample = T([v[bootstrapInds] for v in input])
		outputSample = output[bootstrapInds]
		println(string("Creating tree for bootstrap sample ", i))
		(bootstrapInds, makeRandomSubspaceTree(inputSample, outputSample, p = p))
	end
	for i in 1:B]
end

function makeRandomForest(input::Vector{T}, output::U; p = 10, t = 1e-4, n = length(output), B = 100, seed = 1234) where T <: Any where U <: outputNumType
	sqOutput = output.^2
	l = length(output)
	Random.seed!(seed)
	[begin
		bootstrapInds = ceil.(Int64, rand(n)*l)
	    inputSample = [v[bootstrapInds] for v in input]
		outputSample = output[bootstrapInds]
		sqOutputSample = sqOutput[bootstrapInds]
		println(string("Creating tree for bootstrap sample ", i))
		(bootstrapInds, makeRandomSubspaceTree(inputSample, outputSample, sqOutputSample, p = p, t = t))
	end
	for i in 1:B]
end

function makeRandomForest(input::vectorInputType, output::outputBoolType, testInput, testOutput::outputBoolType; p = 10, n = length(output), B = 100, seed = 1234, prune = true)
	l = length(output)
	Random.seed!(seed)
	origForest = [begin
		bootstrapInds = ceil.(Int64, rand(n)*l)
	    inputSample = [v[bootstrapInds] for v in input]
		outputSample = output[bootstrapInds]
		println(string("Creating tree for bootstrap sample ", i))
		(bootstrapInds, prune ? pruneDecisionTree(makeRandomSubspaceTree(inputSample, outputSample, p = p), testInput, testOutput)[1] : makeRandomSubspaceTree(inputSample, outputSample, p = p))
	end
	for i in 1:B]

	forestMultiOut = [[predictTree(tree[2], testInput[i])[1] for i in 1:length(testInput)] for tree in origForest]
	forestErrs = calcMultiErrs(forestMultiOut, testOutput)
	(combinedForestErr, minIndex) = findmin(forestErrs)
	forestOutput = [predictForest(origForest[1:minIndex], testInput[i])[1] for i in 1:length(testInput)]
	(origForest, forestErrs, minIndex, forestOutput)
end

function makeRandomForest(input::Vector{T}, output::U, testInput, testOutput::U; p = 10, t = 1e-4, n = length(output), B = 100, seed = 1234, prune = true, sortedInd = [eltype(v) <: AbstractFloat ? sortperm(v) : Vector{Int64}() for v in input]) where T <: Any where U <: outputNumType
	sqOutput = output.^2
	sqTestOutput = testOutput.^2
	l = length(output)
	Random.seed!(seed)
	origForest = [begin
		bootstrapInds = if n >= l
			ceil.(Int64, rand(n)*l)
		else
			shuffle(1:l)[1:n]
		end
		println(string("Creating tree for bootstrap sample ", i))
		if prune
			tree = makeRandomSubspaceTree(input, output, sqOutput, testInput, testOutput, sqTestOutput, p = p, t=t, selectedInd=bootstrapInds, sortedInd=sortedInd)
			prunedTree = pruneTestTrainDecisionTree(tree)[1]
			(bootstrapInds, prunedTree)
		else
			tree = makeRandomSubspaceTree(input, output, sqOutput, p = p, t=t, selectedInd=bootstrapInds, sortedInd=sortedInd)
			(bootstrapInds, tree)
		end 
	end
	for i in 1:B]

	forestMultiOutTrain = [runSimplePredictTree(tree[2], input) for tree in origForest]
	forestMultiOutTest = if typeof(origForest[1][2]) <: DecisionTestTrainTree
		[formOrderedTestTrainTreeOutput(tree[2])[2] for tree in origForest]
	else
		[runSimplePredictTree(tree[2], testInput) for tree in origForest]
	end
	forestTrainErrs = calcMultiErrs(forestMultiOutTrain, output)
	forestTestErrs = calcMultiErrs(forestMultiOutTest, testOutput)
	(combinedForestTestErr, minIndex) = findmin(forestTestErrs)
	forestTrainOutput = reduce((a, b) -> a.+b, forestMultiOutTrain[1:minIndex])/minIndex #predictForest(origForest[1:minIndex], testInput)
	forestTestOutput = reduce((a, b) -> a.+b, forestMultiOutTest[1:minIndex])/minIndex # predictForest(origForest[1:minIndex], input)
	(origForest, forestTrainErrs, forestTestErrs, minIndex, forestTrainOutput, forestTestOutput)
end

function makeExtraRandomForest(input::decisionForest, output::T; p = 10, n = length(output), B = 100, seed = 1234) where T <: DenseArray{Bool, 1}
	l = length(output)
	Random.seed!(seed)
	[begin
		bootstrapInds = ceil.(Int64, rand(n)*l)
	    inputSample = [v[bootstrapInds] for v in input]
		outputSample = output[bootstrapInds]
		println(string("Creating tree for bootstrap sample ", i))
		(bootstrapInds, makeExtraTree(inputSample, outputSample, p = p))
	end
	for i in 1:B]
end

