#pragma once

#include <array>
#include <cstddef>
#include <cassert>

#include <algorithm>
#include <span>
#include <ranges>

/*
TODO:

1.  restructure to exclude algorithm and ranges.
    these acount for most of the compile time.
2.  a.  separate runtime-facing constevals
        from intermediate constevals
    b.  separate metavar types from
        intermediate types
    c.  use more convenient and memory efficient
        containers for intermediates?
3.  create proper concepts for Tensor
    and TensorExpression instead of this
    tag system
4.  benchmark always_inline vs inline only
5.  test code correctness with various
    container types and views
6.  ensure correct const awareness
7.  document the damn code
8.  add tensor slicing by giving constexpr integer index
9.  add explicit component retrieval by tensor index
    by (subscript op?)/(index funct?)/(call op?)
10. add sub tensors by using subordinant indices,
    e.g. grab the spatial components of a spacetime
    tensor just by indexing with special indices
11. add matrix inverse and determinant?
12. change tensor template to allow direct user access
    and easy use with template argument deduction guide
13. add more and better user facing
    blueprint/tensor generators
14. try custom literal type for effectively dynamic lists
15. change evaluation to improve -O0 optimized code gen
16. try to add compile time expression optimization?
17. a.  add "void apply(fn)" to tensor which applies
        fn to each component
	b.  add "auto applyAndWrap(fn)" to tensor which
	    applies fn to each component and wraps the
		results in a *new* tensor and returns this
	c.  add these to expressions? for slicing behaviors
18. add stand alone evaluate function
    which returns new function
*/

#define YATTL_INLINE constexpr
// #define YATTL_INLINE inline __attribute__((always_inline))

#define TAG_CONCEPT(TagName)\
namespace Helpers{\
struct TagName{};}\
template <typename T>\
concept C_##TagName = std::is_same_v<typename T::_Tag_##TagName, Helpers::TagName>;

#define ADD_TAG(TagName)\
using _Tag_##TagName = Concepts::Helpers::TagName;

namespace yattl
{
namespace MetaHelpers
{

template <size_t n, typename... Ts>
YATTL_INLINE decltype(auto) get(Ts && ... ts) {
	return std::get<n>(std::forward_as_tuple(ts...));
}

template <size_t N>
consteval std::array<size_t, N> locateKeys(
	auto && keys,
	auto && searchSpace)
{
	std::array<size_t, N> output;
	// searchSpace will be partially ordered, so we will
	// treat it as if sorted but with cyclic boundaries.
	// so whrn we reach the end of searchSpace, we return to the start
	size_t searchIndex = 0;
	auto incr = [&searchIndex, &searchSpace]{searchIndex = (searchIndex + 1) % searchSpace.size();};
	for (size_t n = 0; n < N; ++n)
	{
		// if we come back to here then its not present
		size_t noneFound = searchIndex;
		do // must leave start pos first
		{
			if (keys[n] == searchSpace[searchIndex])
			{
				output[n] = searchIndex;
				incr();
				noneFound = -1;
				break;
			}
			incr();
		}
		while (searchIndex != noneFound);
		assert(noneFound == -1);
		// if (noneFound != -1)
		// 	output[n] = -1;
	}
	return output;
}

template <size_t... is>
YATTL_INLINE  decltype(auto) _helper_expandAndApply(
	auto f, auto... args, auto rangeArgs, std::index_sequence<is...>)
{
	return f(args..., rangeArgs.begin()[is]...);
}

template <size_t n>
YATTL_INLINE  decltype(auto) expandAndApply(auto f, auto... args, auto rangeArgs)
{
	return _helper_expandAndApply(f, args..., rangeArgs, std::make_index_sequence<n>{});
}

}

namespace Index
{
using NameType = char;
using DimType = size_t;
// using NameType = char16_t;

struct IndexFamily
{
	NameType name;
	NameType contractionPairName;
	DimType dim;

	consteval std::strong_ordering operator<=>(IndexFamily const & other) const
	{
		if (name <=> other.name != std::strong_ordering::equal)
			return name <=> other.name;
		if (contractionPairName <=> other.contractionPairName != std::strong_ordering::equal)
			return contractionPairName <=> other.contractionPairName;
		return dim <=> other.dim;
	}
	consteval bool operator==(IndexFamily const & other) const
	{
		return *this <=> other == 0;
	}
};

consteval bool contracts(IndexFamily a, IndexFamily b)
{
	return a.name == b.contractionPairName;
}

struct Index
{
	NameType name;
	IndexFamily family;
	consteval std::strong_ordering operator<=>(Index const & other) const
	{
		if (name <=> other.name != std::strong_ordering::equal)
			return name <=> other.name;
		return family <=> other.family;
	}
	consteval bool operator==(Index const & other) const
	{
		return *this <=> other == 0;
	}
};

consteval bool contracts(Index a, Index b)
{
	return a.name == b.name && contracts(a.family, b.family);
}
consteval bool conflicts(Index a, Index b)
{
	return a.name == b.name && !contracts(a.family, b.family);
}

template <typename T>
consteval bool setwiseEqual(std::span<T> a, std::span<T> b)
{
	// assume sorted
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); ++i)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}
template <typename T>
consteval bool disjoint(std::span<T> a, std::span<T> b)
{
	// assume sorted
	size_t j = 0;
	for (size_t i = 0; i < a.size(); ++i)
	{
		for (; j < b.size(); ++j)
		{
			if (b[j] > a[i]) break;
			if (b[j] < a[i]) continue;
			return false;
		}
	}
	return true;
}

// check that expr1 * expr2 has only
// contracting repeats
consteval bool validRepeats(std::span<Index const> A, std::span<Index const> B)
{
	for (auto&& a : A)
	{
		for (auto&& b : B)
		{
			if (conflicts(a, b))
				return false;
		}
	}
	return true;
}

// check that index_set1 and index_set2 have no
// repeat labels at all
consteval bool noCollisions(std::span<Index const> A, std::span<Index const> B)
{
	for (auto&& a : A)
	{
		for (auto&& b : B)
		{
			if (a.name == b.name)
				return false;
		}
	}
	return true;
}
} // namespace Index

namespace Expression
{
enum class ExpressionType
{
	Atomic,
	Multiply,
	Add,
	Subtract
};

// we could stop tracking indices once contracted,
// but this could give weird bugs since triple
// indices would be missed. So we keep track
// of all free indices, any contractions for
// products and atomics, and all "trash" indices,
// which are all previously contracted indices.
// also a structure like this allows the size
// to be easily known in template one-liners and
// the count of free and contracted indices are
// handled via consteval functions
template <size_t totalSize>
// template <size_t free, size_t contracted, size_t trash>
struct IndexInfo
{
	// std::array<Index, free> freeIndices;
	// std::array<Index, contracted> contractedIndices;
	// std::array<Index, trash> trashIndices;

	std::array<Index::Index, totalSize> allIndices;
	size_t contractedStart;
	size_t trashStart;
	// size_t redundantStart;

	consteval std::span<Index::Index> freeIndices()
	{
		return std::span(
			allIndices.begin(),
			allIndices.begin() + contractedStart);
			// allIndices.data() + contractedStart);
	}
	consteval std::span<Index::Index> contractedIndices()
	{
		return std::span(
			allIndices.begin() + contractedStart,
			allIndices.begin() + trashStart);
	}
	consteval std::span<Index::Index> trashIndices()
	{
		return std::span(
			allIndices.begin() + trashStart,
			allIndices.begin() + totalSize);
			// allIndices.begin() + redundantStart);
	}
	// consteval std::span<Index::Index> redundantIndices()
	// {
	// 	return std::span(
	// 		allIndices.begin() + redundantStart,
	// 		allIndices.begin() + totalSize);
	// }

	consteval std::span<Index::Index const> freeIndices() const
	{
		return std::span(
			allIndices.begin(),
			allIndices.begin() + contractedStart);
			// allIndices.data() + contractedStart);
	}
	consteval std::span<Index::Index const> contractedIndices() const
	{
		return std::span(
			allIndices.begin() + contractedStart,
			allIndices.begin() + trashStart);
	}
	consteval std::span<Index::Index const> trashIndices() const
	{
		return std::span(
			allIndices.begin() + trashStart,
			allIndices.begin() + totalSize);
			// allIndices.begin() + redundantStart);
	}
	// consteval std::span<Index::Index const> redundantIndices() const
	// {
	// 	return std::span(
	// 		allIndices.begin() + redundantStart,
	// 		allIndices.begin() + totalSize);
	// }
};

}

namespace Tensor
{
using PhaseType = int;

template <size_t  n>
using SymmetricGroupElement = std::array<size_t, n>;

template <size_t n>
struct Cycle : std::array<size_t, n>{};

template <size_t n, typename T>
consteval auto applyPermutation(
	SymmetricGroupElement<n> const & permutation,
	std::array<T, n> const & object)
{
	std::array<T, n> output;
	for (int i = 0; i < n; ++i)
	{
		output[permutation[i]] = object[i];
	}
	return output;
}

template <typename T1, typename... Others>
consteval auto composePermutations(T1 const & p1, Others const & ... others)
{
	if constexpr (sizeof...(Others) == 0)
	{
		return p1;
	}
	else
	{
		return applyPermutation(p1, composePermutations(others...));
	}
}

template <typename T1, typename T2, typename... Others>
consteval auto composePermutations(T1 p1, T2 p2, Others... others)
{
	return applyPermutation(p1, composePermutations(p2, others...));
}

template <size_t rank, size_t... ns>
consteval auto makePermutation(
	Cycle<ns>... cycles)
{
	if constexpr (sizeof...(ns) == 0)
	{
		Tensor::SymmetricGroupElement<rank> output;
		for (size_t i = 0; i < rank; ++i)
		{
			output[i] = i;
		}
		return output;
	}
	else if constexpr (sizeof...(ns) == 1)
	{
		Tensor::SymmetricGroupElement<rank> output;
		for (size_t i = 0; i < rank; ++i)
		{
			output[i] = i;
		}
		for (size_t i = 1; i < (ns,...); ++i)
		{
			output[(cycles[i - 1],...)] = (cycles[i],...);
		}
		output[(cycles.back(),...)] = (cycles.front(),...);
		return output;
	}
	else
	{
		return Tensor::composePermutations(makePermutation<rank>(cycles)...);
	}
}

template <size_t rank>
struct IndexSymmetryElement
{
	SymmetricGroupElement<rank> permutationMap;
	int multiplier; // potentially make this rational number for root of unity
};

namespace Helpers
{
consteval size_t pow(size_t base, size_t power)
{
	if (power == 0) return 1;
	return base * pow(base, power - 1);
}

consteval size_t factorial(size_t n)
{
	if (n == 0) return 1;
	return n * factorial(n - 1);
}

// template <size_t rank>
// consteval auto canonicalize(std::array<size_t, rank> const & indices)
}

template <Index::DimType... _dims>
struct StaticElementMap
{
	static constexpr std::array dims{_dims...};
	static constexpr size_t rank = sizeof...(_dims);
	static constexpr size_t size = (_dims * ...);

	bool nonTrivialSymmetry;
	size_t trueSize;
	std::array<std::array<size_t, rank>, size> dataIndexToTensorIndex; // trueSize
	std::array<size_t, size> tensorIndexToDataIndex; // size
	std::array<int, size> tensorIndexToDataPrefactor; // size
};

namespace Helpers
{
template <size_t rank>
consteval std::array<size_t, rank> getStrides(std::array<size_t, rank> dims)
{
	size_t s = 1;
	std::array<size_t, rank> output;
	for (size_t n = rank; n != 0; --n)
	{
		output[n - 1] = s;
		s *= dims[n - 1];
	}
	return output;
}
template <auto dims>
consteval auto decomposeSuperIndex(size_t si)
{
	constexpr auto strides = getStrides(dims);
	constexpr size_t rank = strides.size();
	std::array<size_t, rank> output;
	for (size_t i = 0; i < rank; ++i)
	{
		output[i] = si / strides[i] % dims[i];
	}
	return output;
}
template <auto dims>
consteval auto composeSuperIndex(auto && is)
{
	constexpr auto strides = getStrides(dims);
	constexpr size_t rank = strides.size();
	size_t output = 0;
	for (size_t i = 0; i < rank; ++i)
	{
		output += is[i] * strides[i];
	}
	return output;
}
template <size_t... dims, size_t n>
consteval void treeFill(
	StaticElementMap<dims...> & mapper,
	std::array<IndexSymmetryElement<sizeof...(dims)>, n> const & generators,
	std::array<size_t, sizeof...(dims)> const & indices,
	size_t dataIndex,
	int prefactor)
{
	constexpr size_t rank = sizeof...(dims);
	constexpr auto arrayDims = std::array{dims...};
	for (auto && gen : generators)
	{
		auto newIndices = applyPermutation(gen.permutationMap, indices);
		auto newSuperIndex = composeSuperIndex<arrayDims>(newIndices);
		auto previousPrefactor = mapper.tensorIndexToDataPrefactor[newSuperIndex];
		// check if already zero filled
		if (previousPrefactor == 0) break;
		auto newPrefactor = prefactor * gen.multiplier;
		// check if already consistently filled
		if (previousPrefactor == newPrefactor) continue;
		// check if this has been visited before
		if (previousPrefactor != 2 && prefactor != 0)
		{
			// inconsistency found, remove degree
			// of freedom and zero fill
			--mapper.trueSize;
			treeFill(mapper, generators, indices, dataIndex, 0);
			break;
		}
		// unvisitted, normal fill (maybe part of zero filling)
		mapper.tensorIndexToDataIndex[newSuperIndex] = dataIndex;
		mapper.tensorIndexToDataPrefactor[newSuperIndex] = newPrefactor;
		treeFill(mapper, generators, newIndices, dataIndex, newPrefactor);
	}
}
template <size_t... dims, size_t n>
consteval auto constructElementMap(
	std::array<IndexSymmetryElement<sizeof...(dims)>, n> const & generators)
{
	using SEM = StaticElementMap<dims...>;
	constexpr auto arrayDims = std::array{dims...};
	SEM output{.nonTrivialSymmetry = generators.size() != 0, .trueSize = 0};
	for (size_t i = 0; i < SEM::size; ++i)
	{
		output.tensorIndexToDataPrefactor[i] = 2; // indicates not visited
	}
	for (size_t i = 0; i < SEM::size; ++i)
	{
		if (output.tensorIndexToDataPrefactor[i] != 2)
		{
			continue;
		}
		output.tensorIndexToDataPrefactor[i] = 1;
		output.tensorIndexToDataIndex[i] = output.trueSize;
		output.dataIndexToTensorIndex[output.trueSize] =
			decomposeSuperIndex<arrayDims>(i);
		++output.trueSize;
		treeFill(output, generators, output.dataIndexToTensorIndex[output.trueSize-1],
			output.trueSize-1, 1);
	}
	
	return output;
}

template <typename R>
using RangeElementType = std::remove_reference_t<decltype(*std::declval<R>().begin())>;
}

template <Index::DimType... dims>
struct TensorStructure
{
	// using DataRange = _DataRange;
	// using ComponentType = Helpers::RangeElementType<DataRange>;
	static constexpr size_t rank = sizeof...(dims);

	std::array<Index::IndexFamily, rank> indexStructure;
	StaticElementMap<dims...> elementMap;
};

}

namespace ConstexprWrappers
{
template <auto _value>
struct Generic
{
	static constexpr auto value = _value;
};

template <template<auto...> typename Template, typename Type>
struct TemplateAutoSpec : std::false_type{};
template <template<auto...> typename Template, auto... args>
struct TemplateAutoSpec<Template, Template<args...>> : std::true_type{};

template <template <auto...> typename Template, typename Type>
concept C_TemplateAutoSpec = TemplateAutoSpec<Template, Type>::value;

template <typename T>
concept C_Generic = C_TemplateAutoSpec<Generic, T>;

template <typename T, typename S>
concept C_TypedGeneric = C_Generic<T> && std::is_same_v<std::remove_cvref_t<decltype(T::value)>, S>;

template <typename T>
concept C_Name = C_TypedGeneric<T, Index::NameType>;
template <typename T>
concept C_Dim = C_TypedGeneric<T, Index::DimType>;
template <typename T>
concept C_Family = C_TypedGeneric<T, Index::IndexFamily>;
template <typename T>
concept C_Index = C_TypedGeneric<T, Index::Index>;

template <typename T, template <auto...> typename TT>
concept C_TemplateTypedGeneric = C_Generic<T> && C_TemplateAutoSpec<TT, std::remove_cvref_t<decltype(T::value)>>;

template <typename T>
concept C_Permutation = C_TemplateTypedGeneric<T, Tensor::SymmetricGroupElement>;
template <typename T>
concept C_IndexSymmetryElement = C_TemplateTypedGeneric<T, Tensor::IndexSymmetryElement>;
template <typename T>
concept C_TensorBlueprint = C_TemplateTypedGeneric<T, Tensor::TensorStructure>;

template <typename T>
concept C_Cycle = C_TemplateTypedGeneric<T, Tensor::Cycle>;

template <ConstexprWrappers::C_Family... Families>
struct MetaBlueprint
{
	template <ConstexprWrappers::C_IndexSymmetryElement... Generators>
	YATTL_INLINE auto operator()(Generators...)
	{
		return ConstexprWrappers::Generic<Tensor::TensorStructure<Families::value.dim...>{
			std::array{Families::value...},
			Tensor::Helpers::constructElementMap<Families::value.dim...>(
				std::array{Generators::value...})}>{};
	}
};

}

namespace Concepts
{
namespace Helpers
{
template <std::ranges::range R>
using RangeElementType = std::remove_reference_t<decltype(*std::declval<R>().begin())>;
}

// decides what type can multiply a tensor
template <typename Scalar, typename Component>
concept C_ScalarMultiplyCompatable =
	std::convertible_to<decltype(std::declval<Scalar>() * std::declval<Component>()), Component>;
// decides what type can add to a tensor component
// used for special operator+ with scalar-like tensor expressions
template <typename Scalar, typename Component>
concept C_ScalarAddCompatable =
	std::convertible_to<decltype(std::declval<Scalar>() + std::declval<Component>()), Component>;
// decides what type can compose a tensor.
template <typename T>
concept C_ComponentType =
	std::default_initializable<T> &&
	requires (T a, T b)
	{
		// addition / subtraction
		a = a + b;
		a += b;
		a = a - b;
		a -= b;
		// multiplication
		a = a * b;
		a *= b;
	};

// std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept C_DataRange =
	std::ranges::random_access_range<T> &&
	std::ranges::sized_range<T> &&
	std::ranges::viewable_range<T> &&
	C_ComponentType<Helpers::RangeElementType<T>>;

TAG_CONCEPT(Tensor)
// TAG_CONCEPT(IndexedTensor)

TAG_CONCEPT(TensorExpression)

// // to be scalar addition, one must be scalar-like (zero free indices)
// // and the other must be a scalar
// template <typename Left, typename Right>
// concept ScalarAddCompatable =
// 	Concepts::C_TensorExpression<Left> &&
// 	(Left::indexInfo.freeIndices.size() == 0) &&
// 	yattl::Concepts::<Right>;
// tensor addition must have two tensors with equal free indices
template <typename Left, typename Right>
concept TensorAddCompatable =
	C_TensorExpression<Left> &&
	C_TensorExpression<Right> &&
	Index::setwiseEqual(Left::indexInfo.freeIndices(), Right::indexInfo.freeIndices());

template <typename Left, typename Right>
concept AddCompatable =
	TensorAddCompatable<Left, Right>;

// // to be scalar multiplication, one must be a scalar and the other a tensor
// template <typename Left, typename Right>
// concept ScalarMultiplyCompatable =
// 	Concepts::C_TensorExpression<Left> &&
// 	yattl::Tensor::Concepts::C_ScalarType<Right>;
// to be a tensor product, both must be tensors and
// they must have compatable indices. This requires
// free indices to have only contracting repeats,
// and no repeats may be present with either of the
// trash index sets
template <typename Left, typename Right>
concept TensorMultiplyCompatable =
	C_TensorExpression<Left> &&
	C_TensorExpression<Right> &&
	Index::validRepeats(Left::indexInfo.freeIndices(), Right::indexInfo.freeIndices()) &&
	Index::disjoint(Left::indexInfo.trashIndices(), Right::indexInfo.trashIndices()) &&
	Index::disjoint(Left::indexInfo.freeIndices(), Right::indexInfo.trashIndices()) &&
	Index::disjoint(Left::indexInfo.trashIndices(), Right::indexInfo.freeIndices());

template <typename Left, typename Right>
concept MultiplyCompatable =
	// ScalarMultiplyCompatable<Left, Right> ||
	// ScalarMultiplyCompatable<Right, Left> ||
	TensorMultiplyCompatable<Left, Right>;

template <typename Left, typename Right>
concept AssignmentCompatable =
	C_TensorExpression<Left> &&
	(Left::indexInfo.contractedIndices().size() == 0) &&
	TensorAddCompatable<Left, Right>;

namespace Helpers
{
template <size_t n>
consteval bool sameFamilySets(
	std::array<Index::IndexFamily, n> const & left,
	std::array<Index::IndexFamily, n> const & right)
{
	return Index::setwiseEqual(std::span(left.begin(), n), std::span(right.begin(), n));
}

template <size_t n>
consteval bool allLessThan(yattl::Tensor::Cycle<n> const & c, size_t upperBound)
{
	for (auto && element : c)
	{
		if (element >= upperBound) return false;
	}
	return true;
}

template <size_t n>
consteval bool noRepeats(yattl::Tensor::Cycle<n> c)
{
	std::ranges::sort(c);
	for (size_t i = 1; i < c.size(); ++i)
	{
		if (c[i] == c[i - 1]) return false;
	}
	return true;
}
}
template <size_t rank, std::array<Index::IndexFamily, rank> fams, typename... IndexTypes>
concept ValidIndexing =
	(sizeof...(IndexTypes) == rank) &&
	((ConstexprWrappers::C_Index<IndexTypes> && ...)) &&
	// Helpers::ArrayEqual<Index::IndexFamily, rank, fams, std::array{IndexTypes::index.family...}>::value;
	Helpers::sameFamilySets(fams, std::array{IndexTypes::value.family...});

template <size_t rank, typename T>
concept RankConsistentCycle =
	ConstexprWrappers::C_Cycle<T> &&
	T::value.size() >= 2 &&
	Helpers::allLessThan(T::value, rank) &&
	Helpers::noRepeats(T::value);

template <typename T>
concept C_NonTrivialBlueprint =
	ConstexprWrappers::C_TensorBlueprint<T> && 
	(!T::value.elementMap.nonTrivialSymmetry || T::value.elementMap.trueSize != 0);

}

namespace Evaluate
{
template <Index::Index _index>
struct realizedIndex
{
	static constexpr auto index = _index;
	size_t indexValue;
};

template <typename Expr, Index::Index... indices>
YATTL_INLINE decltype(auto) acquireValue(Expr && expr, realizedIndex<indices>... is);

namespace Helpers
{

template <
auto projectionMap,
typename Expr,
size_t... is,
Index::Index... indices>
YATTL_INLINE decltype(auto) _helper_acquireProjectionValue(
	Expr && expr, std::index_sequence<is...>, realizedIndex<indices>... realIs)
{
	return acquireValue(std::forward<Expr>(expr), MetaHelpers::get<projectionMap[is]>(realIs...)...);
}

template <typename Expr, Index::Index... indices>
YATTL_INLINE decltype(auto) acquireProductProjectionValue(Expr && expr, realizedIndex<indices>... is)
{
	constexpr std::array<Index::Index, sizeof...(indices)> indexSet{indices...};
	constexpr auto leftProjectionMap = MetaHelpers::locateKeys<
		decltype(expr.left)::indexInfo.freeIndices().size()>(
			decltype(expr.left)::indexInfo.freeIndices(),
			std::span(indexSet.begin(), indexSet.end()));
	constexpr auto rightProjectionMap = MetaHelpers::locateKeys<
		decltype(expr.right)::indexInfo.freeIndices().size()>(
			decltype(expr.right)::indexInfo.freeIndices(),
			std::span(indexSet.begin(), indexSet.end()));
	return
		_helper_acquireProjectionValue<leftProjectionMap>(
			expr.left,
			std::make_index_sequence<leftProjectionMap.size()>{},
			is...) *
		_helper_acquireProjectionValue<rightProjectionMap>(
			expr.right,
			std::make_index_sequence<rightProjectionMap.size()>{},
			is...);
}

template <size_t contractionIndex>//, typename T, typename LeftExpr, typename RightExpr>
YATTL_INLINE void addToProduct(auto & result, auto && prodExpr, auto... is)
// YATTL_INLINE void acquireValue(T & result, LeftExpr leftExpr, RightExpr rightExpr)
{
	using Expr = std::remove_reference_t<decltype(prodExpr)>;
	if constexpr (2 * contractionIndex != Expr::indexInfo.contractedIndices().size())
	for (size_t i = 0; i < Expr::indexInfo.contractedIndices()[2 * contractionIndex].family.dim; ++i)
	{
		addToProduct<contractionIndex + 1>(
			result,
			std::forward<decltype(prodExpr)>(prodExpr),
			is...,
			realizedIndex<Expr::indexInfo.contractedIndices()[2 * contractionIndex]>{i},
			realizedIndex<Expr::indexInfo.contractedIndices()[2 * contractionIndex + 1]>{i});
	}
	else
	result += acquireProductProjectionValue(prodExpr, is...);
}

template <size_t N>
consteval std::array<size_t, N> getStrides(std::array<Index::Index, N> is)
{
	size_t s = 1;
	std::array<size_t, N> output;
	for (size_t n = N; n != 0; --n)
	{
		output[n - 1] = s;
		s *= is[n - 1].family.dim;
	}
	return output;
}

consteval size_t eval(size_t in)
{
	return in;
}

template <size_t... is, Index::Index... indices>
YATTL_INLINE size_t constructTrueIndex(
	std::index_sequence<is...>, realizedIndex<indices>... realIs)
{
	constexpr auto strides = getStrides(std::array{indices...});
	return ((eval(strides[is]) * MetaHelpers::get<is>(realIs.indexValue...)) + ...);
}

template <
auto projectionMap,
typename Expr,
size_t... is,
Index::Index... indices>
YATTL_INLINE decltype(auto) _helper_acquireAtomValue(
	Expr && expr, std::index_sequence<is...> ints, realizedIndex<indices>... realIs)
{
	using TrueExpr = std::remove_reference_t<Expr>;
	if constexpr (TrueExpr::TType::elementMap.nonTrivialSymmetry)
	{
		auto index = constructTrueIndex(ints, MetaHelpers::get<projectionMap[is]>(realIs...)...);
		return expr.tensorDataView.begin()[TrueExpr::TType::elementMap.tensorIndexToDataIndex[index]] *
			TrueExpr::TType::elementMap.tensorIndexToDataPrefactor[index];
	}
	else
	{
		return expr.tensorDataView.begin()[
			constructTrueIndex(ints, MetaHelpers::get<projectionMap[is]>(realIs...)...)];
	}
}

template <typename Expr, Index::Index... indices>
YATTL_INLINE decltype(auto) acquireAtomProjectionValue(Expr && expr, realizedIndex<indices>... is)
{
	using TrueExpr = std::remove_reference_t<Expr>;
	constexpr std::array<Index::Index, sizeof...(indices)> indexSet{indices...};
	constexpr auto indexToSlotMap = MetaHelpers::locateKeys<TrueExpr::unsortedIndices.size()>(
		std::span(TrueExpr::unsortedIndices.begin(), TrueExpr::unsortedIndices.end()),
		std::span(indexSet.begin(), indexSet.end()));
	return _helper_acquireAtomValue<indexToSlotMap>(
		std::forward<Expr>(expr),
		std::make_index_sequence<indexToSlotMap.size()>{},
		is...);
}

template <size_t contractionIndex>//, typename T, typename LeftExpr, typename RightExpr>
YATTL_INLINE void addToTrace(auto & result, auto && atomExpr, auto... is)
// YATTL_INLINE void acquireValue(T & result, LeftExpr leftExpr, RightExpr rightExpr)
{
	using Expr = decltype(atomExpr);
	if constexpr (2 * contractionIndex != Expr::indexInfo.contractedIndices().size())
	for (size_t i = 0; i < Expr::indexInfo.contractedIndices()[2 * contractionIndex].family.dim; ++i)
	{
		addToTrace<contractionIndex + 1>(
			result,
			std::forward<decltyp(atomExpr)>(atomExpr),
			is...,
			realizedIndex<Expr::indexInfo.contractedIndices()[2 * contractionIndex]>{i},
			realizedIndex<Expr::indexInfo.contractedIndices()[2 * contractionIndex + 1]>{i});
	}
	else
	result += acquireAtomProjectionValue(atomExpr, is...);
}

}

template <typename Expr, Index::Index... indices>
YATTL_INLINE decltype(auto) acquireValue(Expr && expr, realizedIndex<indices>... is)
{
	using TrueExpr = std::remove_reference_t<Expr>;
	if constexpr (TrueExpr::type == Expression::ExpressionType::Add)
	{
		return acquireValue(expr.left, is...) + acquireValue(expr.right, is...);
	}
	if constexpr (TrueExpr::type == Expression::ExpressionType::Subtract)
	{
		return acquireValue(expr.left, is...) - acquireValue(expr.right, is...);
	}
	if constexpr (TrueExpr::type == Expression::ExpressionType::Multiply)
	{
		if constexpr (TrueExpr::indexInfo.contractedIndices().size() == 0)
		{
			return Helpers::acquireProductProjectionValue(std::forward<Expr>(expr), is...);
		}
		else
		{
			using T = typename TrueExpr::ComponentType;
			T result{};
			Helpers::addToProduct<0>(result, std::forward<Expr>(expr), is...);
			return result;
		}
	}
	if constexpr (TrueExpr::type == Expression::ExpressionType::Atomic)
	{
		if constexpr (TrueExpr::indexInfo.contractedIndices().size() == 0)
		{
			return Helpers::acquireAtomProjectionValue(std::forward<Expr>(expr), is...);
		}
		else
		{
			using T = typename TrueExpr::ComponentType;
			T result{};
			Helpers::addToTrace<0>(result, std::forward<Expr>(expr), is...);
			return result;
		}
	}
}

template <Expression::ExpressionType type, Index::Index... indices>
YATTL_INLINE void trivialMapAssign(auto & destination, auto && source, realizedIndex<indices>... is)
{
	constexpr size_t slot = sizeof...(indices);
	using Expr = std::remove_reference_t<decltype(destination)>;
	if constexpr (slot == Expr::unsortedIndices.size())
	{
		if constexpr (type == Expression::ExpressionType::Atomic)
			Helpers::acquireAtomProjectionValue(destination, is...) =
				acquireValue(std::forward<decltype(source)>(source), is...);
		if constexpr (type == Expression::ExpressionType::Add)
			Helpers::acquireAtomProjectionValue(destination, is...) +=
				acquireValue(std::forward<decltype(source)>(source), is...);
		if constexpr (type == Expression::ExpressionType::Subtract)
			Helpers::acquireAtomProjectionValue(destination, is...) -=
				acquireValue(std::forward<decltype(source)>(source), is...);
		return;
	}
	else
	{
		constexpr auto nextIndex = Expr::indexInfo.freeIndices()[slot];
		for (size_t i = 0; i < nextIndex.family.dim; ++i)
		{
			trivialMapAssign<type>(
				destination,
				std::forward<decltype(source)>(source),
				is...,
				realizedIndex<nextIndex>{i});
		}
	}

}

template <size_t... is>
YATTL_INLINE decltype(auto) _helper_forwardToAcquire(
	auto && expr, auto const & trueIs, std::index_sequence<is...>)
{
	using Expr = std::remove_reference_t<decltype(expr)>;
	return acquireValue(std::forward<decltype(expr)>(expr), realizedIndex<
		Expr::indexInfo.freeIndices()[is]>{trueIs[is]}...);
}

template <size_t n>
YATTL_INLINE decltype(auto) forwardToAcquire(auto && expr, auto const & is)
{
	return _helper_forwardToAcquire(
		std::forward<decltype(expr)>(expr), is, std::make_index_sequence<n>{});
}

template <Expression::ExpressionType type>
YATTL_INLINE void nonTrivialMapAssign(auto & destination, auto && source)
{
	using Expr = std::remove_reference_t<decltype(destination)>;
	for (size_t dataIndex = 0; dataIndex < Expr::TType::elementMap.trueSize; ++dataIndex)
	{
		destination.tensorDataView.begin()[dataIndex] =
			forwardToAcquire<Expr::TType::elementMap.rank>(
				std::forward<decltype(source)>(source),
				Expr::TType::elementMap.dataIndexToTensorIndex[dataIndex]);
	}
}

}

namespace Expression
{
namespace Helpers
{
template <size_t N>
consteval void fullSort(IndexInfo<N> & info)
{
	std::ranges::sort(info.freeIndices());
	std::ranges::sort(info.contractedIndices());
	std::ranges::sort(info.trashIndices());
	// std::ranges::sort(info.redundantIndices());
}

template <size_t N, size_t M>
consteval IndexInfo<N + M> fullMerge(IndexInfo<N> const & left, IndexInfo<M> const & right)
{
	IndexInfo<N + M> output;
	output.contractedStart = left.contractedStart + right.contractedStart;
	output.trashStart = left.trashStart + right.trashStart;
	// output.redundantStart = left.redundantStart + right.redundantStart;
	std::ranges::merge(
		left.freeIndices(),
		right.freeIndices(),
		output.freeIndices().begin());
	std::ranges::merge(
		left.contractedIndices(),
		right.contractedIndices(),
		output.contractedIndices().begin());
	std::ranges::merge(
		left.trashIndices(),
		right.trashIndices(),
		output.trashIndices().begin());
	// std::ranges::merge(
	// 	left.redundantIndices(),
	// 	right.redundantIndices(),
	// 	output.redundantIndices().begin());
	return output;
}

template <size_t N>
consteval void locateAndPlaceContractions(IndexInfo<N> & info)
{
	// use swaps to move contractions to the right spots
	auto freeSpan = info.freeIndices();
	auto newEnd = freeSpan.end();
	for (int i = freeSpan.size() - 1; i > 0; --i)
	{
		if (!Index::contracts(freeSpan[i], freeSpan[i - 1]))
			continue;
		
		--newEnd;
		std::swap(freeSpan[i], *newEnd);

		--newEnd;
		--i;
		std::swap(freeSpan[i], *newEnd);
	}
	info.contractedStart = newEnd - freeSpan.begin();
	// swaps destroyed the sorting of the free indices
	std::ranges::sort(info.freeIndices());
}
template <size_t N>
consteval auto makeAtomicInfo(std::array<Index::Index, N> const & unsortedIndices)
{
	IndexInfo<N> output{
		.allIndices{unsortedIndices},
		.contractedStart = N,
		.trashStart = N,
		// .redundantStart = N
	};
	std::ranges::sort(output.allIndices);
	locateAndPlaceContractions(output);
	return output;
}
template <size_t N, size_t M>
consteval IndexInfo<N + M> joinProductInfos(
	IndexInfo<N> const & left,
	IndexInfo<M> const & right)
{
	IndexInfo<N + M> output = fullMerge(left, right);

	// move previous contractions to the trash
	// std::inplace_merge & ranges variant are not
	// constexpr, so rather than roll out our own
	// we simply sort the full region.
	std::ranges::sort(
		output.allIndices.begin() + output.contractedStart,
		output.allIndices.end());
		// output.allIndices.begin() + output.redundantStart);
	output.trashStart = output.contractedStart;

	locateAndPlaceContractions(output);

	return output;
}
template <size_t N, size_t M>
consteval IndexInfo<N + M> joinAddedInfos(
	IndexInfo<N> const & left,
	IndexInfo<M> const & right)
{
	IndexInfo<N + M> output{
		.allIndices{},
		.contractedStart = left.contractedStart,
		.trashStart = left.contractedStart};

	for (size_t i = 0; i < N; ++i)
	{
		output.allIndices[i] = left.allIndices[i];
	}
	for (size_t i = 0; i < M; ++i)
	{
		output.allIndices[i + N] = right.allIndices[i];
	}
	
	std::ranges::sort(output.trashIndices());
	return output;
}
template <size_t N, size_t M>
consteval IndexInfo<N + M> joinInfos(
	ExpressionType type,
	IndexInfo<N> const & left,
	IndexInfo<M> const & right)
{
	if (type == ExpressionType::Multiply)
	{
		return joinProductInfos(left, right);
	}
	else
	{
		return joinAddedInfos(left, right);
	}
}

template <typename A, typename B>
using MultiplyType = decltype(std::declval<A>() * std::declval<B>());
template <typename A, typename B>
using AddType = decltype(std::declval<A>() + std::declval<B>());
template <ExpressionType type, typename A, typename B>
using ResultantComponentType =
	std::conditional<type == ExpressionType::Multiply, MultiplyType<A, B>, AddType<A, B>>::type;

}

template <ExpressionType _type, typename LeftExpressionType, typename RightExpressionType>
struct TensorExpression // composite (non-atomic) expression
{
	ADD_TAG(TensorExpression)

	using ComponentType = Helpers::ResultantComponentType<
		_type,
		typename LeftExpressionType::ComponentType,
		typename RightExpressionType::ComponentType>;

	static constexpr auto indexInfo = Helpers::joinInfos(
		_type,
		LeftExpressionType::indexInfo,
		RightExpressionType::indexInfo);
	static constexpr ExpressionType type = _type;

	LeftExpressionType left;
	RightExpressionType right;
};

template <typename TensorType, typename IndexArray>
struct TensorExpression<ExpressionType::Atomic, TensorType, IndexArray>
{
	ADD_TAG(TensorExpression)

	using DataView = std::views::all_t<
		typename std::add_lvalue_reference_t<typename TensorType::DataRange>>;
	using ComponentType = typename TensorType::ComponentType;
	using TType = TensorType;

	static constexpr auto unsortedIndices = IndexArray::value;
	static constexpr auto indexInfo = Helpers::makeAtomicInfo(unsortedIndices);
	static constexpr ExpressionType type = ExpressionType::Atomic;

	DataView tensorDataView;
	// TensorExpression(TensorType const & _indexedTensor) : indexedTensor(_indexedTensor){}

	template <typename Other>
		requires Concepts::AssignmentCompatable<TensorExpression, Other>
	YATTL_INLINE TensorExpression & operator=(Other const & other)
	{
		if constexpr (TensorType::elementMap.nonTrivialSymmetry)
		{
			Evaluate::nonTrivialMapAssign<
				ExpressionType::Atomic>(*this, other);
		}
		else
		{
			Evaluate::trivialMapAssign<ExpressionType::Atomic>(*this, other);
		}
		return *this;
	}

	template <typename Other>
		requires Concepts::AssignmentCompatable<TensorExpression, Other>
	YATTL_INLINE TensorExpression & operator+=(Other const & other)
	{
		if constexpr (TensorType::elementMap.nonTrivialSymmetry)
		{
			Evaluate::nonTrivialMapAssign<
				ExpressionType::Add>(*this, other);
		}
		else
		{
			Evaluate::trivialMapAssign<ExpressionType::Add>(*this, other);
		}
		return *this;
	}

	template <typename Other>
		requires Concepts::AssignmentCompatable<TensorExpression, Other>
	YATTL_INLINE TensorExpression & operator-=(Other const & other)
	{
		if constexpr (TensorType::elementMap.nonTrivialSymmetry)
		{
			Evaluate::nonTrivialMapAssign<
				ExpressionType::Subtract>(*this, other);
		}
		else
		{
			Evaluate::trivialMapAssign<ExpressionType::Subtract>(*this, other);
		}
		return *this;
	}
};

template <typename LeftType, typename RightType>
	requires Concepts::AddCompatable<LeftType, RightType>
YATTL_INLINE auto operator+(LeftType && left, RightType && right)
{
	// if constexpr (Concepts::ScalarAddCompatable<RightType, LeftType>) // return a scalar
	// 	return TensorExpression<ExpressionType::Add, RightType, LeftType>{right, left};
	return TensorExpression<
		ExpressionType::Add,
		std::remove_reference_t<LeftType>,
		std::remove_reference_t<RightType>>{left, right};
}

template <typename LeftType, typename RightType>
	requires Concepts::AddCompatable<LeftType, RightType>
YATTL_INLINE auto operator-(LeftType && left, RightType && right)
{
	// if constexpr (Concepts::ScalarAddCompatable<RightType, LeftType>) // return a scalar
	// 	return TensorExpression<ExpressionType::Subtract, RightType, LeftType>{right, left};
	return TensorExpression<
		ExpressionType::Subtract,
		std::remove_reference_t<LeftType>,
		std::remove_reference_t<RightType>>{left, right};
}

template <typename LeftType, typename RightType>
	requires Concepts::MultiplyCompatable<LeftType, RightType>
YATTL_INLINE auto operator*(LeftType && left, RightType && right)
{
	// if constexpr (Concepts::ScalarMultiplyCompatable<RightType, LeftType>)
	// 	return TensorExpression<ExpressionType::Multiply, RightType, LeftType>{right, left};
	// check if we can return a scalar?
	return TensorExpression<
		ExpressionType::Multiply,
		std::remove_reference_t<LeftType>,
		std::remove_reference_t<RightType>>{left, right};
}

} // namespace Expression

namespace Tensor
{
template <auto tensorStructure, typename _DataRange>
class Tensor
{
	public:
	ADD_TAG(Tensor)

	// using DataRange = typename decltype(tensorStructure)::DataRange;
	// using ComponentType = typename decltype(tensorStructure)::ComponentType;
	using DataRange = _DataRange;
	using ComponentType = Helpers::RangeElementType<DataRange>;
	static constexpr size_t rank = tensorStructure.rank;
	static constexpr auto elementMap = tensorStructure.elementMap;
	private:
	DataRange data;

	public:

	Tensor(DataRange const & _data):
		data(_data){}
	Tensor(DataRange && _data):
		data(_data){}

	template <typename... IndexTypes>
		requires Concepts::ValidIndexing<
			rank, tensorStructure.indexStructure, IndexTypes...>
	auto operator()(IndexTypes... indices)
	{
		using IndexArray = std::array<Index::Index, rank>;
		return Expression::TensorExpression<
			Expression::ExpressionType::Atomic,
			Tensor,
			std::integral_constant<
				IndexArray,
				IndexArray{IndexTypes::value...}>>{std::views::all(data)};
	}

	auto dataView()
	{
		return std::views::all(data);
	}
};

} // namespace Tensor

}

namespace yattl
{
template <Index::NameType _name>
consteval auto name()
{
	return ConstexprWrappers::Generic<_name>{};
}
template <Index::NameType... _names>
consteval auto batchNames()
{
	return std::make_tuple(ConstexprWrappers::Generic<_names>{}...);
}
template <Index::DimType _dim>
consteval auto dim()
{
	return ConstexprWrappers::Generic<_dim>{};
}

template <
	ConstexprWrappers::C_Dim Dim,
	ConstexprWrappers::C_Name Name1,
	ConstexprWrappers::C_Name Name2>
YATTL_INLINE auto familyPair(Dim, Name1, Name2)
{
	return std::make_tuple(
		ConstexprWrappers::Generic<Index::IndexFamily{
			.name = Name1::value,
			.contractionPairName = Name2::value,
			.dim = Dim::value}>{},
		ConstexprWrappers::Generic<Index::IndexFamily{
			.name = Name2::value,
			.contractionPairName = Name1::value,
			.dim = Dim::value}>{});
}
template <
	ConstexprWrappers::C_Dim Dim,
	ConstexprWrappers::C_Name Name>
YATTL_INLINE auto selfContractingFamily(Dim, Name)
{
	return ConstexprWrappers::Generic<Index::IndexFamily{
		.name = Name::value,
		.contractionPairName = Name::value,
		.dim = Dim::value}>{};
}

template <
	ConstexprWrappers::C_Family Family,
	ConstexprWrappers::C_Name Name>
YATTL_INLINE auto index(Family, Name)
{
	return ConstexprWrappers::Generic<Index::Index{
		.name = Name::value,
		.family = Family::value}>{};
}

template <
	ConstexprWrappers::C_Family Family,
	ConstexprWrappers::C_Name... Names>
YATTL_INLINE auto batchIndices(Family family, Names... names)
{
	return std::make_tuple(index(family, names)...);
}

template <
	ConstexprWrappers::C_Family Family,
	ConstexprWrappers::C_Name... Names>
YATTL_INLINE auto batchPipeIndices(Family family, std::tuple<Names...>)
{
	return std::make_tuple(index(family, Names{})...);
}

template <
	Concepts::C_DataRange DataRange,
	ConstexprWrappers::C_Family... Families>
YATTL_INLINE auto basicTensor(DataRange data, Families...)
{
	constexpr Tensor::TensorStructure<Families::value.dim...> structure{
		.indexStructure{Families::value...},
		.elementMap{.nonTrivialSymmetry = false}};
	return Tensor::Tensor<structure, DataRange>(std::move(data));
}

template <size_t... cycleSlots>
consteval auto cycle()
{
	return ConstexprWrappers::Generic<Tensor::Cycle<sizeof...(cycleSlots)>{cycleSlots...}>{};
}

template <size_t rank, int multiplier, typename... Cycles>
	requires (
		(Concepts::RankConsistentCycle<rank, Cycles> && ...) &&
		(multiplier == 1 || multiplier == -1))
consteval auto symmetryGenerator(Cycles...)
{
	return ConstexprWrappers::Generic<
		Tensor::IndexSymmetryElement<rank>{
			Tensor::makePermutation<rank>(Cycles::value...),
			multiplier}>{};
}

template <
	ConstexprWrappers::C_Family... Families>
YATTL_INLINE auto tensorBlueprint(Families...)
{
	return ConstexprWrappers::MetaBlueprint<Families...>{};
}

template <ConstexprWrappers::C_Family... Families>
YATTL_INLINE auto basicBlueprint(Families...)
{
	return ConstexprWrappers::Generic<
		Tensor::TensorStructure<Families::value.dim...>{
			.indexStructure{Families::value...},
			.elementMap{.nonTrivialSymmetry = false}}>{};
}

template <
	Concepts::C_DataRange DataRange,
	Concepts::C_NonTrivialBlueprint Blueprint>
YATTL_INLINE auto tensor(DataRange r, Blueprint)
{
	return Tensor::Tensor<Blueprint::value, DataRange>(std::move(r));
}

} // namespace yattl
