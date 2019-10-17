abstract type DecisionTree{T} end
abstract type DecisionTestTrainTree{T} <: DecisionTree{T} end

struct DecisionLeaf{T} <: DecisionTree{T}
	ind::Any
	prediction::T
	err::T
end

struct DecisionTestTrainLeaf{T} <: DecisionTestTrainTree{T}
	ind::Any
	prediction::T
	err::T
	testInd::Vector{Int64}
	testErr::T
end

struct DecisionNode{T} <: DecisionTree{T}
	leaf::DecisionLeaf{T}
	splitCol::Int64
	splitPoint::T
	left::DecisionTree{T}
	right::DecisionTree{T}
end

struct DecisionTestTrainNode{T} <: DecisionTestTrainTree{T}
	leaf::DecisionTestTrainLeaf{T}
	splitCol::Int64
	splitPoint::T
	left::DecisionTree{T}
	right::DecisionTree{T}
end

struct DecisionNullNode{T} <: DecisionTree{T}
	leaf::DecisionLeaf{T}
	splitCol::Int64
	splitPoint::T
	nullLeft::Bool
	left::DecisionTree{T}
	right::DecisionTree{T}
end

struct DecisionLabelNode{T} <: DecisionTree{T}
	leaf::DecisionLeaf{T}
	splitCol::Int64
	leftPartition::Set{Int64}
	left::DecisionTree{T}
	right::DecisionTree{T}
end

struct DecisionTestTrainLabelNode{T} <: DecisionTestTrainTree{T}
	leaf::DecisionTestTrainLeaf{T}
	splitCol::Int64
	leftPartition::Set{Int64}
	left::DecisionTree{T}
	right::DecisionTree{T}
end

struct DecisionNullLabelNode{T} <: DecisionTree{T}
	leaf::DecisionLeaf{T}
	splitCol::Int64
	leftPartition::Set{Int64}
	nullLeft::Bool
	left::DecisionTree{T}
	right::DecisionTree{T}
end

abstract type DecisionGraph{T} end
struct DecisionGraphNode{T} <: DecisionGraph{T}
	prediction::T
	splitCol::Int64
	splitPoint::T
	left::DecisionGraph{T}
	right::DecisionGraph{T}
end
struct DecisionGraphLabelNode{T} <: DecisionGraph{T}
	prediction::T
	splitCol::Int64
	leftPartition::Set{Int64}
	left::DecisionGraph{T}
	right::DecisionGraph{T}
end
struct DecisionGraphTerminus{T} <: DecisionGraph{T}
	prediction::T
end

DecisionBranch = Union{DecisionNode, DecisionLabelNode, DecisionTestTrainNode, DecisionTestTrainLabelNode}
DecisionTerm = Union{DecisionLeaf, DecisionTestTrainLeaf}

ruleTypes = Union{Set{Int64}, Tuple{T, T} where T <: AbstractFloat}
singleInputType = DenseArray{T, 1} where T <: Real
singleNullInputType = DenseArray{T, 1} where T <: Union{U, Missing} where U <: Real
floatInputType = DenseArray{T, 1} where T <: AbstractFloat
labInputType = DenseArray{Int64, 1}
floatNullInputType = DenseArray{T, 1} where T <: Union{U, Missing} where U <: AbstractFloat
labNullInputType = DenseArray{T, 1} where T <: Union{Int64, Missing}
vectorInputType = DenseArray{T, 1} where T <: singleInputType 
vectorNullInputType = DenseArray{T, 1} where T <: singleNullInputType 
outputBoolType = DenseArray{Bool, 1}
outputNumType = DenseArray{T, 1} where T <: Real
inputType2 = Array{Array{T, 1}, 1} where T
decisionForest = Vector{Tuple{Vector{Int64}, T}} where T <: DecisionTree