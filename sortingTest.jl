# selectedInd::AbstractArray{Int64, 1}
# input::Vector{T} where T <: AbstractFloat
# output::Vector{T} where T <: AbstractFloat

# sortedInd = sortperm(view(input, selectedInd))

# #can I quickly get sortedInd if I have input presorted when it is passed to the function?
# #let's assume I have sortedInputInds::AbstractArray{Int64, 1}

# input[sortedInputInds] #would be input sorted
# input[selectedInd] #is the set i'm working with

# #I can map selectedInd to their position in the sortedInput vector
# inputSortDict = Dict(zip(1:length(input), sortedInputInds)) #mapping from original input ind to sorted index
# invInputSortDict = Dict(zip(sortedInputInds, 1:length(input))) #mapping from input in original order to sorted order

# tmp = [invInputSortDict[a] for a in selectedInd] #this will give the order of selectedInd as they exist in the sorted input vector

#what i want is a way to iterate through selectedInd in their sorted order

#Assume we have a vector V of Float64 that is unsorted. If we sort the entire vector at the start and then generate 
#a sampled set of indices for that vector, can that sample be directly put into order from the original sorting vector?

V = rand(10)
sortedInd = sortperm(V)
selectedInd = [5, 3, 7]
selectedIndDict = Dict(zip(1:length(selectedInd), selectedInd))
inputSortDict = Dict(zip(1:length(V), sortedInd))
invInputSortDict = Dict(zip(sortedInd, 1:length(V)))
tmp1 = [invInputSortDict[a] for a in selectedInd]
tmp2 = [inputSortDict[a] for a in selectedInd]
#if i use sortperm on tmp1 I will get the same result as sortperm on V[selectedInd] but then I still have to sort it
#rather that directly get sortperm(selectedInd)

#let's say selectedInd consists of 5, 2.  When I get element 5 from V, I know where 5 belongs in the sorted vector so in this case 
#element 5 belongs in the 3rd position.  Element 2 belongs in the 9th position.  But in the sampled vector there are only two
#elements so the ordering here would be 5, 2 because 3 is less than 9.  So what I need is a mapping that takes 3, 9 and maps it to
#1, 2.  5 -> 3 -> 1 & 2 -> 9 -> 2.

V[filter(in(a, selectedInd), sortedInd])] #here we filter out of sortedInd the selected ones and then pass that

#so if we go through sortedInd and only keep elements that are in selectedInd then the resulting vector will be the selectedInd
#in a sorted order so there will be no need to sort them further.

tmp1 = [invInputSortDict[a] for a in selectedInd]
selectedIndDict = Dict(zip(tmp1, 1:length(tmp1)))

#what I want is to get [5, 7, 3] from [5, 3, 7], I know this will work if I simply filter through sortedInd as follows:
orderedSelectedInd = filter(a -> in(a, selectedInd), sortedInd) #but an I do this without filtering?

#another option is to build up the ordered list from tmp1 which in this case is [3, 10, 6] so we know that 5 goes in position
#3, 3 goes in position 10, and 7 goes in position 6 but our final vector is only of length 3.  

#=

Vector      sorting	sorted	 	selected  
elements	order 	position	inds
v1 			4		3			x
v2			2		2
v3			1		4			x
v4			3		1
v5			5		5			x                 




so I want elements v1, v3, v5 which should be sorted as v3, v1, v5.  So what if I had a pre-sorted vector, I would need to know
which selected indices to pick out that were not the original ones of [1, 3, 5].  If V is sorted and I select indicies out of it
that are also sorted then the resulting vector will still be sorted.  This would require making sure that the selected indices are
always presented as sorted which itself would be a sorting operation for every split that isn't any reduction in work.  What if 
the selected indices are always kept as a boolean vector of length equal to the original vector that simply indicates whether an 
index is used or not?  In that case it would be trivial to generate the list of indices themselves but would this make sorting
the sampled vector easier?  It would ensure that the generated selection is always in order in the context of the original vector.
So then we can take V[sortedInd][newSelectedInd] where newSelectedInd is the ordered list of selected indices being sampled mapped
to their positoin in the sorting vector.  So in this example I want indices [1, 3, 5] but in the sorting vector that would map to 
indices [3, 4, 5].  So if I had V sorted with [v4, v2, v1, v3, v5] to get the proper selection out of this I would need to select
indices[3, 4, 5] from the sorted vector so can I map [1, 3, 5] to [3, 4, 5]?  Yes, take selectedInd as a boolean and sort it
by the sorting index so that would go from [1, 0, 1, 0, 1] to [0, 0, 1, 1, 1] and then take the true elements of this in order which
would result in [3, 4, 5].  So we have for the indices in order sortedInd[find(selectedInd[sortedInd])]
=#
V = rand(10)
selectedInd = rand(Bool, 10)
sortedInd = sortperm(V)
V[sortedInd[find(selectedInd[sortedInd])]]


