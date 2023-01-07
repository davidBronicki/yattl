#include "yattl.hpp"

#include <vector>
#include <iostream>

int main()
{

	// // spacetype up and down families
	// auto [up, down] = yattl::familyPair(
	// 	yattl::dim<4>(), yattl::name<'u'>(), yattl::name<'d'>());
	// spatial family
	auto spatial = yattl::selfContractingFamily(yattl::dim<3>(), yattl::name<'s'>());

	auto [_i, _j, _k, _l, _m, _n, _a, _b, _c, _d] =
		yattl::batchNames<'i', 'j', 'k', 'l', 'm', 'n', 'a', 'b', 'c', 'd'>();
	
	auto [i, j, k, l, m, n] = yattl::batchIndices(spatial, _i, _j, _k, _l, _m, _n);
	// auto [a, b, c, d] = yattl::indices(down, _a, _b, _c, _d);
	// auto [A, B, C, D] = yattl::indices(up, _a, _b, _c, _d);

	auto vectorBlueprint = yattl::basicBlueprint(spatial);
	auto matrixBlueprint = yattl::tensorBlueprint(spatial, spatial)(
		yattl::symmetryGenerator<2, 1>(yattl::cycle<0,1>()));
	// auto matrixBlueprint = yattl::basicBlueprint(spatial, spatial);

	std::cout << matrixBlueprint.value.elementMap.trueSize << "\n";
	for (auto && item : matrixBlueprint.value.elementMap.dataIndexToTensorIndex)
	{
		for (auto && i : item)
		{
			std::cout << i << ", ";
		}
		std::cout << "\n";
	}

	std::vector<double> testData{3, 1, 0};
	// std::vector<double> tData2{3,6,3,3,9,6,3,3,6};
	std::vector<double> symData{3,6,3,3,9,6};

	// auto testTensor = yattl::basicTensor(testData, spatial);
	// auto testTensor2 = yattl::basicTensor(tData2, spatial, spatial);

	auto testTensor = yattl::tensor(testData, vectorBlueprint);
	// auto testTensor2 = yattl::tensor(tData2, matrixBlueprint);
	auto testTensor2 = yattl::tensor(symData, matrixBlueprint);

	testTensor2(i, j) = testTensor(i) * testTensor(j);
	// volatile double thing = testTensor2.dataView()[0];
	auto expr = testTensor2(i, j);
	// auto expr2 = testTensor(i) * testTensor(j);
	// expr = expr2;
	// auto expr = testTensor(i) * testTensor2(i, j) * testTensor(k) +
	// 	testTensor2(k, j) * testTensor2(i, i);

	auto info = testTensor2.dataView();
	for (auto && item : info)
	{
		std::cout << item << "\n";
	}

	std::cout << "\nfree:\n";
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
	// std::cout << "\nredundant:\n";
	// for (auto && index : decltype(expr)::indexInfo.redundantIndices())
	// {
	// 	std::cout << index.name;
	// }
	std::cout << "\n";

	// constexpr auto val = decltype(i)::index == decltype(j)::index;
}
