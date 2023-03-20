#include "../yattl.hpp"

#include <vector>
#include <iostream>

int main()
{

	// spacetype up and down families
	auto [up, down] = yattl::familyPair(
		yattl::dim<4>(), yattl::name<'u'>(), yattl::name<'d'>());
	// spatial family
	auto spatial = yattl::selfContractingFamily(yattl::dim<3>(), yattl::name<'s'>());

	// create a bunch of names. The important information is held in the types.
	auto [_i, _j, _k, _l, _m, _n, _a, _b, _c, _d] =
		yattl::batchNames<'i', 'j', 'k', 'l', 'm', 'n', 'a', 'b', 'c', 'd'>();
	
	// create indices. Again, important info held in the types.

	// "spatial" indices, or for "internal degrees of freedom"
	auto [i, j, k, l, m, n] = yattl::batchIndices(spatial, _i, _j, _k, _l, _m, _n);

	// "spacetime" indices, or "covariant" and "contravariant" respectively
	auto [a, b, c, d] = yattl::batchIndices(down, _a, _b, _c, _d);
	auto [A, B, C, D] = yattl::batchIndices(up, _a, _b, _c, _d);

	// basic vector blueprint. This is for a three dimensional vector since "spatial" is 3-D
	auto vectorBlueprint = yattl::basicBlueprint(spatial);

	// symmetric tensor blueprint. Two spatial indices and a symmetry set.
	// symmetryGenerator<rank, parity>. for antisymmetry, parity=-1.
	// cycle<0,1>() -- when indices 0 and 1 are cycled, you end up with
	// an identical state except with a factor of "parity". That is,
	// T(i, j) == (parity) * T(j, i)
	auto matrixBlueprint = yattl::tensorBlueprint(spatial, spatial)(
		yattl::symmetryGenerator<2, 1>(yattl::cycle<0,1>()));

	// riemann tensor blueprint.
	// R(i,j,k,l) = -R(j,i,k,l)
	// R(i,j,k,l) = -R(i,j,l,k)
	// R(i,j,k,l) =  R(k,l,i,j)
	// These alone give 21 degrees of freedom in four dimensions.
	// The real Riemann tensor has 20 degrees of freedom, but the last
	// constraint requires a type of symmetry which yattl cannot express.
	auto riemBlueprint = yattl::tensorBlueprint(down, down, down, down)(
		yattl::symmetryGenerator<4, -1>(yattl::cycle<0,1>()),
		yattl::symmetryGenerator<4, -1>(yattl::cycle<2,3>()),
		yattl::symmetryGenerator<4, 1>(yattl::cycle<1,3>(), yattl::cycle<0,2>()));

	// example of actually using the blueprints.
	std::vector<double> testData{3, 1, 0};
	// constructors are available for move semantics on the underlying data
	auto testTensor = yattl::tensor(std::move(testData), vectorBlueprint);
	std::cout << "testData was moved into tensor. Size is now " << testData.size() << ".\n"; // NOLINT

	std::vector<double> symData{3,6,3,3,9,6}; // NOLINT
	auto testTensor2 = yattl::tensor(std::move(symData), matrixBlueprint);
	std::cout << "symData was moved into tensor. Size is now " << symData.size() << ".\n"; // NOLINT

	testTensor2(i, j) = testTensor(i) * testTensor(j);

	for (auto && item : testTensor2.dataView())
	{
		std::cout << item << "\n";
	}
	std::cout << "\n";

	// the rest of the program displays constexpr data which is ordinarilly blackbox info.
	// this isn't important for actually *using* the library.

	// this displays the order of elements in memory. e.g., the last element is index (2, 2),
	// but this is not the nineth element in memory, but instead the sixth. This
	// is because it is a symmetric tensor and so we don't have to track three redundant elements.
	std::cout << matrixBlueprint.value.elementMap.trueSize << "\n";
	for (auto && item : matrixBlueprint.value.elementMap.dataIndexToTensorIndex)
	{
		for (auto && i : item)
		{
			std::cout << i << ", ";
		}
		std::cout << "\n";
	}

	// display index order of riemann tensor blueprint. This has a lot of zeros
	// since the symmetry structure is extensive.
	std::cout << "\n" << riemBlueprint.value.elementMap.trueSize << "\n";
	for (auto && item : riemBlueprint.value.elementMap.dataIndexToTensorIndex)
	{
		for (auto && i : item)
		{
			std::cout << i << ", ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	// set up a tensor expression to view the static info on free indices,
	// contracted indices, and "trash" indices.
	// only free indices here.
	// contracted is for indices contracted at top level expression.
	// trash indices for contracted indices in sub-expressions.
	auto expr = testTensor2(i, j);

	std::cout << "free:\n";
	for (auto && index : decltype(expr)::indexInfo.freeIndices())
	{
		std::cout << index.name;
	}
	std::cout << "\ncontracted:\n";
	for (auto && index : decltype(expr)::indexInfo.contractedIndices())
	{
		std::cout << index.name;
	}
	std::cout << "\ntrash:\n";
	for (auto && index : decltype(expr)::indexInfo.trashIndices())
	{
		std::cout << index.name;
	}
	std::cout << "\n";
}
