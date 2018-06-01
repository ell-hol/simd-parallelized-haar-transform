#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <immintrin.h>

//#define NO_INLINE __attribute__((noinline))
//#define ALIGNED16 __attribute__((aligned(16)))

int input[100] __attribute__ ((aligned (16)));
int output[100] __attribute__ ((aligned (16)));


int main() {
	for(int i=0; i<100; i++){
		output[i] = i;
	}

	//scalar code
	for(int i=0; i<100; i++){
		output[i] = output[i]*output[i];
	}

	//intrinsics code
	__m128i *in_vec = (__m128i *) input;	//cast to SIMD vector type
	__m128i *out_vec = (__m128i *) output;
	__m128i xmm0;


	// set a vector to (0x 00000003 00000003 00000003 00000003 )
	__m128i add3 = _mm_set1_epi32(3) ;

	 for (int i=0; i<25; i++) {
	 	xmm0 = _mm_load_si128 (&in_vec [ i ] ) ; 	// load 4 i n t e g e r s
		xmm0 = _mm_mullo_epi32 ( xmm0 , xmm0 ) ; 	// square 4 i n t e g e r s
		xmm0 = _mm_add_epi32 ( xmm0 , add3 ) ;   	// add 4 i n t e g e r s
		_mm_store_si128(&out_vec [ i ] , xmm0 ) ; // s t o r e back 4 i n t e g e r s
	 }


	 printf("%d\n", out_vec[0]);

	return 1;
}
