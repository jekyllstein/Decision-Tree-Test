# function findRegSplit(input::Vector{T}, output::Vector{T}, sqOutput::Vector{T}, sortedInd::AbstractVector{Int64}, invSortedInd::AbstractVector{Int64}, selectedInd::AbstractVector{Bool}) where T <: AbstractFloat
# #finds the split point for input data to minimize the combined variance 
# #of the resulting two output vectors.	
# 	# sortedSelection = sortedInd[findall(view(selectedInd, sortedInd))]
# 	sortedSelection = sortedInd[findall(selectedInd[sortedInd])]
# 	l = length(sortedSelection)
# 	ss = sum(view(sqOutput, sortedSelection)) #sum(sqOutput.*selectedInd) 
# 	s1 = zero(T)
# 	s2 = sum(view(output, sortedSelection)) #sum(output.*selectedInd)
# 	errorSplits = [begin
# 		s1 += output[sortedSelection[i]]
# 		s2 -= output[sortedSelection[i]]
# 		ss - s1*s1/i - s2*s2/(l-i) # sum(output[sortedInd[1:i]])^2/i - sum(output[sortedInd[i+1:l]])^2 / (l-i)
# 	end
# 	for i = 1:l-1]
# 	(minValue, minIndex) = findmin(errorSplits)
# 	(selection1, selection2) = getSplits(sortedSelection, minIndex, length(output))
# 	# println(selection1)
# 	# println(selection2)
# 	splitPoint = input[sortedSelection[minIndex]]
# 	(T(splitPoint), selection1, selection2, T(minValue))
# end 

function findRegSplit(input::Vector{T}, output::Vector{T}, sqOutput::Vector{T}, sortedInd::AbstractVector{Int64}, invSortedInd::AbstractVector{Int64}, selectedInd::AbstractVector{Bool}) where T <: AbstractFloat
#finds the split point for input data to minimize the combined variance 
#of the resulting two output vectors.	
	# sortedSelection = sortedInd[findall(view(selectedInd, sortedInd))]
	# sortedSelection = sortedInd[findall(selectedInd[sortedInd])]
	l = length(sortedInd)
	ss = selectSum(sqOutput, selectedInd) 
	s1 = zero(T)
	c1 = 0
	s2 = selectSum(output, selectedInd)
	c2 = count(selectedInd)
	baseErr = ss - s2*s2/c2

	minValue = baseErr
	minIndex = l
	for i = 1:l
		if selectedInd[sortedInd[i]]
			s1 += output[sortedInd[i]]
			s2 -= output[sortedInd[i]]  
			c1 += 1
			c2 -= 1
			newErr = if c2 == 0
				baseErr
			else
				ss - s1*s1/c1 - s2*s2/c2
			end
			if newErr < minValue
				minValue = newErr
				minIndex = i
			end
		end
	end
	# errorSplits = [begin
	# 	if selectedInd[i]
	# 		s1 += output[i]
	# 		s2 -= output[i]  
	# 		c1 += 1
	# 		c2 -= 1
	# 		# ifelse(c2 == 0, baseErr, ss - s1*s1/c1 - s2*s2/c2)
	# 		if c2 == 0
	# 			baseErr
	# 		else
	# 			ss - s1*s1/c1 - s2*s2/c2
	# 		end
	# 	else
	# 		Inf
	# 	end
	# end
	# for i in sortedInd]
	# (minValue, minErrIndex) = findmin(errorSplits)
	# (minValue, minIndex) = findmin(errorSplits)
	# (minValue, minIndex) = getMinSplit(1, s1, s2, c1, c2, ss, baseErr, sortedInd[end])
	# println(minIndex)
	# println(errorSplits)
	# minIndex = errorInds[minErrIndex]
	# println(ss)
	# println(count(selectedInd))
	(selection1, selection2) = getSplits(selectedInd, sortedInd,  minIndex)
	# println(selection1)
	# println(selection2)
	splitPoint = input[sortedInd[minIndex]]
	(T(splitPoint), selection1, selection2, T(minValue))
end 

function findRegSplit(input::Vector{Int64}, output::Vector{T}, sqOutput::Vector{T}, sortedInd::AbstractVector{Int64}, invSortedInd::AbstractVector{Int64}, selectedInd::AbstractVector{Bool}) where T <: AbstractFloat
#finds the split point for input data to minimize the combined variance 
#of the resulting two output vectors.
	sortedSelection = sortedInd[findall(view(selectedInd, sortedInd))]
	l = length(sortedSelection)
	labelDict = Dict([(a, Vector{T}()) for a in unique(input[sortedSelection])]) 
	indDict = Dict([(a, Vector{Int64}()) for a in unique(input[sortedSelection])]) 
	for i in sortedSelection
		push!(labelDict[input[i]], output[i])
		push!(indDict[input[i]], i)
	end

	#calculate the mean output for every label and sort labels by them
	labelOutputs = [mean(labelDict[a]) for a in keys(labelDict)]
	sortedLabels = collect(keys(labelDict))[sortperm(labelOutputs)]

	s = sum(view(output, sortedSelection))
	ss = sum(view(sqOutput, sortedSelection))
	s1 = 0
	s2 = s
	c1 = 0
	c2 = l
	errorSplits = [begin
		s1 += sum(labelDict[label])
		s2 -= sum(labelDict[label])
		c1 += length(labelDict[label])
		c2 -= length(labelDict[label])
		if c1 == 0
			ss - s2*s2/c2
		elseif c2 == 0
			ss - s1*s1/c1
		else
			ss - s1*s1/c1 - s2*s2/c2
		end
	end
	for label = sortedLabels]
	(minValue, minIndex) = findmin(errorSplits)
	leftPartition = Set(sortedLabels[1:minIndex])
	rightPartition = if minIndex < length(sortedLabels)
		Set(sortedLabels[minIndex+1:end])
	else
		Set{Int64}()
	end
	ind1 = mapreduce(a -> indDict[a], vcat, leftPartition)
	ind2 = isempty(rightPartition) ? Vector{Int64}() : mapreduce(a -> indDict[a], vcat, rightPartition)
	
	(leftPartition, indToBool(ind1, length(output)), indToBool(ind2, length(output)), T(minValue))
end 

function findRegSplit(input::U, output::Vector{T}, sqOutput::Vector{T}, selectedInd::AbstractArray{Int64}, sortedInd::AbstractArray{Int64} = Vector{Int64}()) where T <: AbstractFloat where U <: Vector{V} where V <: AbstractFloat
#finds the split point for input data to minimize the combined variance 
#of the resulting two output vectors.	
	l = length(selectedInd)
	n = length(output)
	# sortedInd = sortperm(view(input, selectedInd))
	usePresort = if !isempty(sortedInd)
		if n < 1000
			true
		elseif n < 1e4
			l > 300
		elseif n < 1e5
			l > 3000
		elseif n < 1e6
			l > 25000
		elseif n < 2e6
			l > 45000
		elseif n < 4e6
			l > 80000
		else
			l > 300000
		end
	else
		false
	end

	sortedSelection = if usePresort
		sortReplacementSample(sortedInd, selectedInd)
		# sortSample(sortedInd, selectedInd)
	else
		view(selectedInd, sortperm(view(input, selectedInd)))
	end
	# println(sortedSelection)
	ss = sum(view(sqOutput, selectedInd))
	s1 = zero(T)
	s2 = sum(view(output, selectedInd))
	minValue = Inf
	minIndex = 0
	for i in 1:l-1
		# s1 += output[selectedInd[sortedInd[i]]]
		s1 += output[sortedSelection[i]]
		# s2 -= output[selectedInd[sortedInd[i]]]
		s2 -= output[sortedSelection[i]]
		newErr = ss - s1*s1/i - s2*s2/(l-i)
		if newErr < minValue
			minValue = newErr
			minIndex = i
		end
	end
	# println(minValue)

	# errorSplits = [begin
	# 	s1 += output[selectedInd[sortedInd[i]]]
	# 	s2 -= output[selectedInd[sortedInd[i]]]
	# 	ss - s1*s1/i - s2*s2/(l-i) # sum(output[sortedInd[1:i]])^2/i - sum(output[sortedInd[i+1:l]])^2 / (l-i)
	# end
	# for i = 1:l-1]
	# (minValue, minIndex) = findmin(errorSplits)
	# ind1 = view(selectedInd, sortedInd)[1:minIndex]
	ind1 = sortedSelection[1:minIndex]
	# ind2 = view(selectedInd, sortedInd)[minIndex+1:l]
	ind2 = sortedSelection[minIndex+1:l]
	# splitPoint = input[selectedInd[sortedInd[minIndex]]]
	splitPoint = input[sortedSelection[minIndex]]
	(T(splitPoint), ind1, ind2, T(minValue))
end 

function findRegSplit(input::AbstractVector{Bool}, output::Vector{T}, sqOutput::Vector{T}, selectedInd::AbstractArray{Int64}) where T <: AbstractFloat
#finds the split point for input data to minimize the combined variance 
#of the resulting two output vectors.	
	set1 = findall(view(input, selectedInd))
	set2 = findall(.!view(input, selectedInd))

	ss = sum(view(sqOutput, selectedInd))
	s1 = sum(view(view(output, selectedInd), set1))
	s2 = sum(view(view(output, selectedInd), set2))
	splitErr = ss - s1*s1/length(set1) - s2*s2/length(set2)
	combinedErr = ss - ((s1+s2)^2)/length(selectedInd)
	# println(string("Split Err: ", splitErr, ", Combined Err: ", combinedErr))
	if splitErr <= combinedErr
		(Set(0), selectedInd[set2], selectedInd[set1], splitErr)
	else
		(Set{Int64}(), Vector{Int64}(), selectedInd, combinedErr)
	end
end 

function findRegSplit(input::Vector{Int64}, output::Vector{T}, sqOutput::Vector{T}, selectedInd::AbstractVector{Int64}) where T <: AbstractFloat 
#finds the split point for input data to minimize the combined entropy (p_true*log2(p_true) + p_false*log2(p_false))
#of the resulting two output vectors. input is a vector of integers representing categorical labels to be split at some
#partition of labels 
	l = length(selectedInd)
	# labels = unique(view(input, selectedInd))
	maxLabel = maximum(view(input, selectedInd))
	labelSums = zeros(T, maxLabel)
	labelCounts = zeros(Int64, maxLabel)
	#create a dictionary mapping each input label in selectedInd to a vector of output labels in selectedInd
	# labelDict = Dict([(a, Vector{T}()) for a in unique(input[selectedInd])]) 
	# indDict = Dict([(a, Vector{Int64}()) for a in unique(input[selectedInd])]) 
	indDict = Dict([(a, Vector{Int64}()) for a in 1:maxLabel]) 
	# indDict = Dict([(a, Set{Int64}()) for a in 1:maxLabel]) #fill(Set{Int64}(), maxLabel)
	for i in selectedInd
		labelSums[input[i]] += output[i]
		labelCounts[input[i]] += 1
		# push!(labelDict[input[i]], output[i])
		push!(indDict[input[i]], i)
		# push!(indLists[input[i]], i)
	end

	#calculate the mean output for every label and sort labels by them
	# labelOutputs = [mean(labelDict[a]) for a in keys(labelDict)]
	labelOutputs = labelSums./labelCounts
	# sortedLabels = keys(labelDict)[sortperm(labelOutputs)]
	sortedLabels = sortperm(labelOutputs)

	s = sum(view(output, selectedInd))
	ss = sum(view(sqOutput, selectedInd))
	s1 = 0
	s2 = s
	c1 = 0
	c2 = l
	minValue = Inf
	minIndex = 0
	leftPartition = Vector{Int64}()
	# ind1 = Vector{Int64}()
	# ind2 = selectedInd
	for i in 1:length(sortedLabels)
		label = sortedLabels[i]
		s1 += labelSums[label]
		s2 -= labelSums[label]
		c1 += labelCounts[label]
		c2 -= labelCounts[label]
		newErr = if c1 == 0
			ss - s2*s2/c2
		elseif c2 == 0
			ss - s1*s1/c1
		else
			ss - s1*s1/c1 - s2*s2/c2
		end
		if newErr < minValue
			minValue = newErr
			minIndex = i
			push!(leftPartition, label) 
			# ind1 = vcat(ind1, indDict[label]) #push!(ind1, indDict[label])
			# ind2 = setdiff(ind2, indDict[label])
		end
	end

	ind1 = if minIndex == 0
		Vector{Int64}()
	elseif minIndex == length(sortedLabels)
		selectedInd
	else
		reduce(vcat, [indDict[sortedLabels[i]] for i in 1:minIndex])
	end

	ind2 = if minIndex == 0
		selectedInd
	elseif minIndex == length(sortedLabels)
		Vector{Int64}()
	else
		reduce(vcat, [indDict[sortedLabels[i]] for i in minIndex+1:length(sortedLabels)]) # setdiff(selectedInd, ind1)
	end

	# ind2 = reduce(vcat, [indDict[sortedLabels[i]] for i in minIndex+1:length(sortedLabels)])

	# errorSplits = [begin
	# 	# s1 += sum(labelDict[label])
	# 	s1 += labelSums[label]
	# 	# s2 -= sum(labelDict[label])
	# 	s2 -= labelSums[label]
	# 	# c1 += length(labelDict[label])
	# 	c1 += labelCounts[label]
	# 	# c2 -= length(labelDict[label])
	# 	c2 -= labelCounts[label]
	# 	if c1 == 0
	# 		ss - s2*s2/c2
	# 	elseif c2 == 0
	# 		ss - s1*s1/c1
	# 	else
	# 		ss - s1*s1/c1 - s2*s2/c2
	# 	end
	# end
	# for label = sortedLabels]
	# (minValue, minIndex) = findmin(errorSplits)
	# leftPartition = Set(sortedLabels[1:minIndex])
	# rightPartition = if minIndex < length(sortedLabels)
	# 	Set(sortedLabels[minIndex+1:end])
	# else
	# 	Set{Int64}()
	# end
	# ind1 = mapreduce(a -> indDict[a], vcat, leftPartition)
	# selectedIndSet = Set(selectedInd)
	# rightIndSet = setdiff(selectedInd, mapreduce(a -> indDict[a], vcat, leftPartition))
	# leftIndSet = setdiff(selectedInd, rightIndSet)
	# ind2 = isempty(rightPartition) ? Vector{Int64}() : mapreduce(a -> indDict[a], vcat, rightPartition)
	# ind2 = isempty(rightPartition) ? Vector{Int64}() : collect(mapreduce(a -> indLists[a], union, rightPartition))
	# ind1 = collect(leftIndSet)
	# ind2 = collect(rightIndSet)
	(Set(leftPartition), ind1, ind2, minValue)
end 

function findBoolSplit(input::floatInputType, output::outputBoolType, sortedInd::AbstractVector{Int64}, invSortedInd::AbstractVector{Int64}, selectedInd::AbstractVector{Bool})
#finds the split point for input data to minimize the combined entropy (p_true*log2(p_true) + p_false*log2(p_false))
#of the resulting two output vectors. input is a vector of values to be split at some numerical point. 
	sortedSelection = sortedInd[findall(selectedInd[sortedInd])] #selectedInd reordered in the sorting order
	l = length(sortedSelection)
	# sortedInd = sortperm(view(input, selectedInd))
	num_true = sum(view(output, sortedSelection))
	n1 = 0
	n2 = num_true
	errorSplits = [begin
		n1 += output[sortedSelection[i]]
		n2 -= output[sortedSelection[i]]
		if (n1 == i) | (n1 == 0)
			if (n2 == l-i) | (n2 == 0)
				0.0
			else
				(- n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/l
			end
		else
			if (n2 == l-i) | (n2 == 0)
				(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i))/l
			else
				(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i) - n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/l  
			end
		end
	end
	for i = 1:l-1]
	(minValue, minIndex) = findmin(errorSplits)
	# (selection1, selection2) = getSplits(selectedInd, sortedInd, invSortedInd, minIndex)
	(selection1, selection2) = getSplits(selectedInd, sortedInd, minIndex)
	splitPoint = input[sortedSelection[minIndex]]
	(AbstractFloat(splitPoint), selection1, selection2, AbstractFloat(minValue))
end 

function findBoolSplit(input::DenseVector{Int64}, output::outputBoolType, sortedInd::AbstractVector{Int64}, invSortedInd::AbstractVector{Int64}, selectedInd::AbstractVector{Bool})
#finds the split point for input data to minimize the combined entropy (p_true*log2(p_true) + p_false*log2(p_false))
#of the resulting two output vectors. input is a vector of values to be split at some numerical point. 
	sortedSelection = sortedInd[findall(selectedInd[sortedInd])] #selectedInd reordered in the sorting order
	l = length(sortedSelection)
	
	labelDict = Dict([(a, Vector{Bool}()) for a in unique(input[sortedSelection])]) 
	indDict = Dict([(a, Vector{Int64}()) for a in unique(input[sortedSelection])]) 
	for i in sortedSelection
		push!(labelDict[input[i]], output[i])
		push!(indDict[input[i]], i)
	end

	#calculate the true fraction for every label and sort labels by them
	labelTrues = [mean(labelDict[a]) for a in keys(labelDict)]
	sortedLabels = collect(keys(labelDict))[sortperm(labelTrues)]

	num_true = sum(view(output, sortedSelection))
	n1 = 0
	n2 = num_true
	c1 = 0
	c2 = l
	errorSplits = [begin
		n1 += sum(labelDict[label])
		n2 -= sum(labelDict[label])
		c1 += length(labelDict[label])
		c2 -= length(labelDict[label])
		f1 = n1/c1
		f2 = n2/c2
		if (n1 == c1) | (n1 == 0)
			if (n2 == c2) | (n2 == 0)
				0.0
			else
				(- n2*log2(n2/c2) - (c2-n2)*log2((c2-n2)/c2))/l
			end
		else
			if (n2 == c2) | (n2 == 0)
				(-n1*log2(f1) - (c1-n1)*log2((c1-n1)/c1))/l
			else
				(-n1*log2(n1/c1) - (c1-n1)*log2((c1-n1)/c1) - n2*log2(n2/c2) - (c2-n2)*log2((c2-n2)/c2))/l  
			end
		end
	end
	for label = sortedLabels]

	(minValue, minIndex) = findmin(errorSplits)
	leftPartition = Set(sortedLabels[1:minIndex])
	rightPartition = if minIndex < length(sortedLabels)
		Set(sortedLabels[minIndex+1:end])
	else
		Set{Int64}()
	end
	ind1 = mapreduce(a -> indDict[a], vcat, leftPartition)
	ind2 = isempty(rightPartition) ? Vector{Int64}() : mapreduce(a -> indDict[a], vcat, rightPartition)
	
	(leftPartition, indToBool(ind1, length(output)), indToBool(ind2, length(output)), AbstractFloat(minValue))
end 

function findBoolSplit(input::T, output::outputBoolType, selectedInd::W) where T <: floatInputType where W <: AbstractArray{Int64, 1}
#finds the split point for input data to minimize the combined entropy (p_true*log2(p_true) + p_false*log2(p_false))
#of the resulting two output vectors. input is a vector of values to be split at some numerical point. 
	l = length(selectedInd)
	sortedInd = sortperm(view(input, selectedInd))
	num_true = sum(view(output, selectedInd))
	n1 = 0
	n2 = num_true
	errorSplits = [begin
		n1 += output[selectedInd][sortedInd[i]]
		n2 -= output[selectedInd][sortedInd[i]]
		if (n1 == i) | (n1 == 0)
			if (n2 == l-i) | (n2 == 0)
				0.0
			else
				(- n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/l
			end
		else
			if (n2 == l-i) | (n2 == 0)
				(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i))/l
			else
				(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i) - n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/l  
			end
		end
	end
	for i = 1:l-1]
	(minValue, minIndex) = findmin(errorSplits)
	ind1 = selectedInd[view(sortedInd, 1:minIndex)]
	ind2 = selectedInd[view(sortedInd, minIndex+1:l)]
	splitPoint = view(input, selectedInd)[sortedInd[minIndex]]
	(AbstractFloat(splitPoint), ind1, ind2, AbstractFloat(minValue))
end 

function findBoolSplit(input::T, output::outputBoolType, selectedInd::W) where T <: floatInputType where W <: AbstractArray{Int64, 1}
#finds the split point for input data to minimize the combined entropy (p_true*log2(p_true) + p_false*log2(p_false))
#of the resulting two output vectors. input is a vector of values to be split at some numerical point. 
	hasValues = (!ismissing(a) for a in view(input, selectedInd))
	nonNullInds = findall(hasValues)
	nullInds = findall(a -> !a, hasValues)
	if isempty(nonNullInds)
		num_true = sum(view(output, selectedInd))
		minValue = if (num_true == 0) | (num_true == length(selectedInd))
			zero(AbstractFloat)
		else
			-num_true*log2(num_true/length(selectedInd)) - (length(selectedInd)-num_true)*log2(length(selectedInd-num_true)/length(selectedInd))/length(selectedInd)
		end
		(missing, false, Vector{Int64}(), selectedInd, AbstractFloat(minValue))
	elseif length(nonNullInds) == length(selectedInd)
		sortedInd = sortperm(view(input, selectedInd))
		l = length(sortedInd)
		num_true = sum(view(output, selectedInd))
		n1 = 0
		n2 = num_true
		errorSplits = [begin
			n1 += output[selectedInd][sortedInd[i]]
			n2 -= output[selectedInd][sortedInd[i]]
			if (n1 == i) | (n1 == 0)
				if (n2 == l-i) | (n2 == 0)
					zero(AbstractFloat)
				else
					(- n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/l
				end
			else
				if (n2 == l-i) | (n2 == 0)
					(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i))/l
				else
					(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i) - n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/l  
				end
			end
		end
		for i = 1:l-1]
		(minValue, minIndex) = findmin(errorSplits)
		ind1 = selectedInd[view(sortedInd, 1:minIndex)]
		ind2 = selectedInd[view(sortedInd, minIndex+1:l)]
		splitPoint = view(input, selectedInd)[sortedInd[minIndex]].value
		(AbstractFloat(splitPoint), false, ind1, ind2, AbstractFloat(minValue))
	elseif length(nonNullInds) == 1
		l = length(selectedInd)
		nonNullTrues = sum(view(view(output, selectedInd), nonNullInds))
		nullTrues = sum(view(view(output, selectedInd), nullInds))
		numNull = length(nullInds)
		numNotNull = 1
		
		(E1, E2) = if ((nonNullTrues + nullTrues) == l) | ((nonNullTrues + nullTrues) == 0)
			(zero(V), zero(V))
		elseif (nullTrues == l-1)
			if nonNullTrues == 1
				(zero(AbstractFloat), zero(AbstractFloat))
			else
				((-(l-1)*log((l-1)/l)-log2(1/l))/100, zero(AbstractFloat))
			end
		else
			E1 = (-(nonNullTrues+nullTrues)*log2((nonNullTrues+nullTrues)/l) - (l-nonNullTrues-nullTrues)*log2((l-nonNullTrues-nullTrues)/l))/l
			E2 = (-(nullTrues)*log2(nullTrues/(l-1)) - (l-1-nullTrues)*log2((l-1-nullTrues)/(l-1)))/(l-1)
			(E1, E2)
		end

		if E1 < E2
			#better to put nulls and non nulls together
			(missing, true, selectedInd, Vector{Int64}(), E1)
		else
			#better to separate nulls and non nulls
			(missing, false, view(selectedInd, nonNullInds), view(selectedInd, nullInds), E2)
		end
	else
		sortedInd = sortperm(view(view(input, selectedInd), nonNullInds))
		l = length(sortedInd)		
		basei = length(nullInds)
		(minValue1, minIndex1) = begin
			num_true = sum(view(view(output, selectedInd), nonNullInds))
			n1 = 0 + sum(view(view(output, selectedInd), nullInds))
			n2 = num_true
			errorSplits = [begin
				n1 += output[selectedInd][nonNullInds][sortedInd[i]]
				n2 -= output[selectedInd][nonNullInds][sortedInd[i]]
				if (n1 == i+basei) | (n1 == 0)
					if (n2 == l-i) | (n2 == 0)
						zero(AbstractFloat)
					else
						(- n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/(l+basei)
					end
				else
					if (n2 == l-i) | (n2 == 0)
						(-n1*log2(n1/(i+basei)) - (basei+i-n1)*log2((basei+i-n1)/(basei+i)))/(l+basei)
					else
						(-n1*log2(n1/(i+basei)) - (basei+i-n1)*log2((basei+i-n1)/(basei+i)) - n2*log2(n2/(l-i)) - (l-i-n2)*log2((l-i-n2)/(l-i)))/(l+basei)  
					end
				end
			end
			for i = 1:l-1]
			findmin(errorSplits)
		end
		
		(minValue2, minIndex2) = begin
			num_true = sum(view(output, selectedInd))
			n1 = 0
			n2 = num_true
			errorSplits = [begin
				n1 += output[selectedInd][nonNullInds][sortedInd[i]]
				n2 -= output[selectedInd][nonNullInds][sortedInd[i]]
				if (n1 == i) | (n1 == 0)
					if (n2 == l-i) | (n2 == 0)
						zero(AbstractFloat)
					else
						(-n2*log2(n2/(l-i+basei)) - (l-i+basei-n2)*log2((l-i+basei-n2)/(l-i+basei)))/(l+basei)
					end
				else
					if (n2 == l-i+basei) | (n2 == 0)
						(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i))/(l+basei)
					else
						(-n1*log2(n1/i) - (i-n1)*log2((i-n1)/i) - n2*log2(n2/(l-i+basei)) - (l-i+basei-n2)*log2((l-i+basei-n2)/(l-i+basei)))/(l+basei)  
					end
				end
			end
			for i = 1:l-1]
			findmin(errorSplits)
		end

		if minValue1 <= minValue2
			ind1 = [selectedInd[nonNullInds][view(sortedInd, 1:minIndex1)]; view(selectedInd, nullInds)]
			ind2 = selectedInd[nonNullInds][view(sortedInd, minIndex1+1:l)]
			splitPoint = view(input, selectedInd)[nonNullInds][sortedInd[minIndex1]].value
			(AbstractFloat(splitPoint), true, ind1, ind2, AbstractFloat(minValue1))
		else
			ind1 = selectedInd[nonNullInds][view(sortedInd, 1:minIndex2)]
			ind2 = [selectedInd[nonNullInds][view(sortedInd, minIndex2+1:l)]; view(selectedInd, nullInds)]
			splitPoint = view(input, selectedInd)[nonNullInds][sortedInd[minIndex2]].value
			(AbstractFloat(splitPoint), false, ind1, ind2, AbstractFloat(minValue2))
		end
	end
end

function findBoolSplit(input::Vector{Int64}, output::outputBoolType, selectedInd::U) where U <: AbstractArray{Int64, 1}
#finds the split point for input data to minimize the combined entropy (p_true*log2(p_true) + p_false*log2(p_false))
#of the resulting two output vectors. input is a vector of integers representing categorical labels to be split at some
#partition of labels 
	l = length(selectedInd)
	#create a dictionary mapping each input label in selectedInd to a vector of output labels in selectedInd
	labelDict = Dict([(a, Vector{Bool}()) for a in unique(input[selectedInd])]) 
	indDict = Dict([(a, Vector{Int64}()) for a in unique(input[selectedInd])]) 
	for i in selectedInd
		push!(labelDict[input[i]], output[i])
		push!(indDict[input[i]], i)
	end

	#calculate the true fraction for every label and sort labels by them
	labelTrues = [mean(labelDict[a]) for a in keys(labelDict)]
	sortedLabels = collect(keys(labelDict))[sortperm(labelTrues)]

	num_true = sum(view(output, selectedInd))
	n1 = 0
	n2 = num_true
	c1 = 0
	c2 = l
	errorSplits = [begin
		n1 += sum(labelDict[label])
		n2 -= sum(labelDict[label])
		c1 += length(labelDict[label])
		c2 -= length(labelDict[label])
		f1 = n1/c1
		f2 = n2/c2
		if (n1 == c1) | (n1 == 0)
			if (n2 == c2) | (n2 == 0)
				0.0
			else
				(- n2*log2(n2/c2) - (c2-n2)*log2((c2-n2)/c2))/l
			end
		else
			if (n2 == c2) | (n2 == 0)
				(-n1*log2(f1) - (c1-n1)*log2((c1-n1)/c1))/l
			else
				(-n1*log2(n1/c1) - (c1-n1)*log2((c1-n1)/c1) - n2*log2(n2/c2) - (c2-n2)*log2((c2-n2)/c2))/l  
			end
		end
	end
	for label = sortedLabels]

	(minValue, minIndex) = findmin(errorSplits)
	leftPartition = Set(sortedLabels[1:minIndex])
	rightPartition = if minIndex < length(sortedLabels)
		Set(sortedLabels[minIndex+1:end])
	else
		Set{Int64}()
	end
	ind1 = mapreduce(a -> indDict[a], vcat, leftPartition)
	ind2 = isempty(rightPartition) ? Vector{Int64}() : mapreduce(a -> indDict[a], vcat, rightPartition)
	
	(leftPartition, ind1, ind2, minValue)
end 

function findBoolSplit(input::Vector{V}, output::outputBoolType, selectedInd::U) where U <: AbstractArray{Int64, 1} where V Union{Int64, Missing}
#finds the split point for input data to minimize the combined entropy (p_true*log2(p_true) + p_false*log2(p_false))
#of the resulting two output vectors. input is a vector of integers representing categorical labels to be split at some
#partition of labels 
	l = length(selectedInd)
	#create a dictionary mapping each input label in selectedInd to a vector of output labels in selectedInd
	labelDict = Dict([(a, Vector{Bool}()) for a in unique(input[selectedInd])]) 
	indDict = Dict([(a, Vector{Int64}()) for a in unique(input[selectedInd])]) 
	for i in selectedInd
		push!(labelDict[input[i]], output[i])
		push!(indDict[input[i]], i)
	end

	#calculate the true fraction for every label and sort labels by them
	labelTrues = [mean(labelDict[a]) for a in keys(labelDict)]
	sortedLabels = collect(keys(labelDict))[sortperm(labelTrues)]

	num_true = sum(view(output, selectedInd))
	n1 = 0
	n2 = num_true
	c1 = 0
	c2 = l
	errorSplits = [begin
		n1 += sum(labelDict[label])
		n2 -= sum(labelDict[label])
		c1 += length(labelDict[label])
		c2 -= length(labelDict[label])
		f1 = n1/c1
		f2 = n2/c2
		if (n1 == c1) | (n1 == 0)
			if (n2 == c2) | (n2 == 0)
				0.0
			else
				(- n2*log2(n2/c2) - (c2-n2)*log2((c2-n2)/c2))/l
			end
		else
			if (n2 == c2) | (n2 == 0)
				(-n1*log2(f1) - (c1-n1)*log2((c1-n1)/c1))/l
			else
				(-n1*log2(n1/c1) - (c1-n1)*log2((c1-n1)/c1) - n2*log2(n2/c2) - (c2-n2)*log2((c2-n2)/c2))/l  
			end
		end
	end
	for label = sortedLabels]

	(minValue, minIndex) = findmin(errorSplits)
	leftPartition = Set(sortedLabels[1:minIndex])
	rightPartition = if minIndex < length(sortedLabels)
		Set(sortedLabels[minIndex+1:end])
	else
		Set{Int64}()
	end
	nullLeft = in(Nullable{Int64}(), leftPartition)

	ind1 = mapreduce(a -> indDict[a], vcat, leftPartition)
	ind2 = isempty(rightPartition) ? Vector{Int64}() : mapreduce(a -> indDict[a], vcat, rightPartition)
	
	filteredLeft = filter(a -> a.hasvalue, leftPartition)
	leftPartition = isempty(filteredLeft) ?  Set{Int64}() : Set([a.value for a in filteredLeft])

	(leftPartition, nullLeft, ind1, ind2, minValue)

end 

function randomBoolSplit(input::Vector{Int64}, output::T, selectedInd::U) where T <: DenseArray{Bool, 1} where U <: AbstractArray{Int64, 1}
#finds a random split point to partition input data. input is a vector of integers representing categorical labels to be split at some partition of labels 
	inputList = unique(view(input, selectedInd))
	leftPartition = Set(inputList[findall(rand(Bool, length(inputList)))])
	ind1 = selectedInd[findall(a -> in(a, leftPartition), view(input, selectedInd))]
	ind2 = selectedInd[findall(a -> !in(a, leftPartition), view(input, selectedInd))]
	E = if isempty(ind1)
		if isempty(ind2)
			0.0
		else
			calcBoolEntropy(view(output, ind2))
		end
	else
		if isempty(ind2)
			calcBoolEntropy(view(output, ind1))
		else
			(calcBoolEntropy(view(output, ind1))*length(ind2) + calcBoolEntropy(view(output, ind1))*length(ind1))/(length(ind1)+length(ind2))
		end
	end
	(leftPartition, ind1, ind2, E)
end 

function randomBoolSplit(input::floatInputType, output::U, selectedInd::W) where T <: AbstractFloat where U <: DenseArray{Bool, 1} where W <: AbstractArray{Int64, 1}
#finds a random split point for input data input is a vector of values to be split at some numerical point. 
	minInput = minimum(input)
	maxInput = maximum(input)
	splitPoint = minInput + (maxInput - minInput)*rand()
	ind1 = selectedInd[findall(a -> a < splitPoint, view(input, selectedInd))]
	ind2 = selectedInd[findall(a -> a >= splitPoint, view(input, selectedInd))]
	E = if isempty(ind1)
		if isempty(ind2)
			0.0
		else
			calcBoolEntropy(view(output, ind2))
		end
	else
		if isempty(ind2)
			calcBoolEntropy(view(output, ind1))
		else
			(calcBoolEntropy(view(output, ind1))*length(ind2) + calcBoolEntropy(view(output, ind1))*length(ind1))/(length(ind1)+length(ind2))
		end
	end
	(T(splitPoint), ind1, ind2, T(E))
end 