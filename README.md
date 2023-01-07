# yattl
Yet Another Tensor Template Library

Yattl is a light weight, header only library for working with
numerical tensor expressions. It uses template expressions to
delay evaluation until an assignment, where it does the full
evaluation inside a single loop (or nested loop). Since the
library is header only, the computation is completely transparent
to the compiler, allowing for optimizations on par with
hand-written loops.

Yattl supports natural indexing with Einstein summation convention,
e.g. instead of writing

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			A[i] = B[i][j] * C[j];
		}
	}

or even writing

	Index<i>(A) = Index<i, j>(B) * Index<j>(C);

you can write

	A(i) = B(i, j) * C(j);

While this is certainly not the first library to support natural
indexing or similar (notable examples being FTensor, Blitz++,
and the tensor library of SpECTRE), it does have some worthwhile
advanatages:
- **c++20 constraints** on user facing templates producing (more) readable
error messages before full template specialization.
- **Light weight, header-only** with only standard library dependencies.
- **Fully templated data representation** allowing extensive freedom for users.
E.g. using a custom data container with custom allocator; using a c++20
std::views object to make a temporary view tensor for quick,
on the spot computation; or using scalar fields for tensor components.
- **Arbitrary (anti-)symmetries on any indices** with zero-waste storage size
and component computing. To the best of
my knowledge, this is the first library with this feature.

Be warned, however: this library is incomplete and the API is likely to change
abruptly. Many features I envision having are not implemented and the code
completely lacks documentation.
