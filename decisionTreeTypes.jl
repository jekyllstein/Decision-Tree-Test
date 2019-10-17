abstract type DecisionTree{T} end
abstract type DecisionTestTrainTree{T} <: DecisionTree{T} end

struct DecisionLeaf{T} <: DecisionTree{T}
	ind::Vector{Int64}
	prediction::T
	err::T
end

struct DecisionTestTrainLeaf{T} <: DecisionTestTrainTree{T}
	ind::Vector{Int64}
	prediction::T
	err::T
	testInd::Vector{Int64}
	testErr::T
end

struct DecisionNode{T} <: DecisionTree{T}
	leaf::DecisionLeaf{T}
	splitCol::Int64
	splitPoint::T
	missingLeft::Bool
	left::DecisionTree{T}
	right::DecisionTree{T}
end

#Default to missing values going left
DecisionNode(leaf::DecisionLeaf{T}, splitCol::Int64, splitPoint::T, left::DecisionTree{T}, right::DecisionTree{T}) where T <:Real = DecisionNode(leaf, splitCol, splitPoint, true, left, right)

struct DecisionTestTrainNode{T} <: DecisionTestTrainTree{T}
	leaf::DecisionTestTrainLeaf{T}
	splitCol::Int64
	splitPoint::T
	missingLeft::Bool
	left::DecisionTree{T}
	right::DecisionTree{T}
end

#Default to missing values going left
DecisionTestTrainNode(leaf::DecisionTestTrainLeaf{T}, splitCol::Int64, splitPoint::T, left::DecisionTree{T}, right::DecisionTree{T}) where T <:Real = DecisionTestTrainNode(leaf, splitCol, splitPoint, true, left, right)

struct DecisionLabelNode{T} <: DecisionTree{T}
	leaf::DecisionLeaf{T}
	splitCol::Int64
	leftPartition::Set{Int64}
	missingLeft::Bool
	left::DecisionTree{T}
	right::DecisionTree{T}
end

#Default to missing values going left
DecisionLabelNode(leaf::DecisionLeaf{T}, splitCol::Int64, leftPartition::Set{Int64}, left::DecisionTree{T}, right::DecisionTree{T}) where T <: Real = DecisionLabelNode(leaf, splitCol, leftPartition, true, left, right)

struct DecisionTestTrainLabelNode{T} <: DecisionTestTrainTree{T}
	leaf::DecisionTestTrainLeaf{T}
	splitCol::Int64
	leftPartition::Set{Int64}
	missingLeft::Bool
	left::DecisionTree{T}
	right::DecisionTree{T}
end

#Default to missing values going left
DecisionTestTrainLabelNode(leaf::DecisionTestTrainLeaf{T}, splitCol::Int64, leftPartition::Set{Int64}, left::DecisionTree{T}, right::DecisionTree{T}) where T <: Real = DecisionTestTrainLabelNode(leaf, splitCol, leftPartition, true, left, right)

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

Branch = Union{DecisionNode, DecisionLabelNode, DecisionTestTrainNode, DecisionTestTrainLabelNode, DecisionGraphNode, DecisionGraphLabelNode}
Term = Union{DecisionLeaf, DecisionTestTrainLeaf, DecisionGraphTerminus}
DecisionBranch = Union{DecisionNode, DecisionLabelNode, DecisionTestTrainNode, DecisionTestTrainLabelNode}
DecisionTerm = Union{DecisionLeaf, DecisionTestTrainLeaf}
LabelBranch = Union{DecisionLabelNode, DecisionTestTrainLabelNode, DecisionGraphLabelNode}
ValueBranch = Union{DecisionNode, DecisionTestTrainNode, DecisionGraphNode}

ruleTypes = Union{Set{Int64}, Tuple{T, T} where T <: AbstractFloat}
singleInputType = DenseVector{T} where T <: Union{Real, Missing}
floatInputType = DenseVector{T} where T <: Union{AbstractFloat, Missing}
labInputType = DenseVector{T} where T <: Union{Int64, Missing}
vectorInputType = DenseVector{T} where T <: singleInputType 
outputBoolType = AbstractVector{Bool}
outputNumType = DenseVector{T} where T <: Real
inputType2 = Vector{Vector{T}} where T
decisionForest = Vector{Tuple{Vector{Int64}, T}} where T <: DecisionTree