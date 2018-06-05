#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <immintrin.h>

#define ROWS 16
#define COLS 16

#define NO_INLINE __attribute__((noinline))
#define ALIGNED16 __attribute__((aligned(16)))

static inline uint8_t avg(uint8_t a, uint8_t b) {
	return (uint8_t) (((uint16_t) a + (uint16_t) b + 1) / 2);
}

static inline void haar_x_scalar(uint8_t *output, const uint8_t *input) {
	for (size_t y = 0; y < ROWS; y++) {
		uint8_t tmp_input_row[COLS];
		memcpy(tmp_input_row, &input[y * COLS], COLS);

		for (size_t lim = COLS; lim > 1; lim /= 2) {
			for (size_t x = 0; x < lim; x += 2) {
				uint8_t a = tmp_input_row[x];
				uint8_t b = tmp_input_row[x+1];
				uint8_t s = avg(a, b);
				uint8_t d = avg(a, -b);
				tmp_input_row[x/2] = s;
				output[y * COLS + (lim+x)/2] = d;
			}
		}

		output[y * COLS] = tmp_input_row[0];
	}
}

static inline void haar_y_scalar(uint8_t *output, const uint8_t *input) {
	for (size_t x = 0; x < COLS; x++) {
		uint8_t tmp_input_col[ROWS];
		for (size_t y = 0; y < ROWS; y++) {
			tmp_input_col[y] = input[y * COLS + x];
		}

		for (size_t lim = COLS; lim > 1; lim /= 2) {
			for (size_t y = 0; y < lim; y += 2) {
				uint8_t a = tmp_input_col[y];
				uint8_t b = tmp_input_col[y+1];
				uint8_t s = avg(a, b);
				uint8_t d = avg(a, -b);
				tmp_input_col[y/2] = s;
				output[(lim+y)/2 * COLS + x] = d;
			}
		}

		output[x] = tmp_input_col[0];
	}
}

NO_INLINE static void haar_scalar(uint8_t *output, const uint8_t *input) {
	uint8_t tmp[ROWS*COLS];

	haar_x_scalar(output, input);

	// //TODO : Orginals, uncomment later following lines and delete previous one
	// haar_x_scalar(tmp, input);
	// haar_y_scalar(output, tmp);
}

static inline void haar_x_simd(uint8_t *output, const uint8_t *input) {
	//haar_x_scalar(output, input);

	// TODO Vectorize me


	__m128i *in_vec = (__m128i *) input;			//casting to SIMD vector type
	__m128i *out_vec = (__m128i *) output;

	__m128i even_mask_128 = _mm_setr_epi8(0x80, 0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0, 2, 4, 6, 8, 10, 12, 14);
	__m128i odd_mask_128 = _mm_setr_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 1, 3, 5, 7, 9, 11, 13, 15);

	__m128i even_mask_64 = _mm_setr_epi8(0x80, 0x80, 0x80, 0x80, 0, 2, 4, 6, 0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80);
	__m128i odd_mask_64 = _mm_setr_epi8(0x80, 0x80, 0x80, 0x80, 1, 3, 5, 7, 0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80);

	__m128i even_mask_32 = _mm_setr_epi8(0x80, 0x80, 0, 2, 0x80, 0x80, 0x80, 0x80, 0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80);
	__m128i odd_mask_32 = _mm_setr_epi8(0x80, 0x80, 1, 3, 0x80, 0x80, 0x80, 0x80, 0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80);

	__m128i even_mask_16 = _mm_setr_epi8(0x80, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80);
	__m128i odd_mask_16 = _mm_setr_epi8(0x80, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80 ,0x80);

	__m128i zero128 = _mm_setzero_si128();

	// Temporaries declaration !
	__m128i avg_plus_128;
	__m128i avg_minus_128;

	__m128i avg_plus_64;
	__m128i avg_minus_64;

	__m128i avg_plus_32;
	__m128i avg_minus_32;

	__m128i avg_plus_16;
	__m128i avg_minus_16;

	__m128i temp_128;

	// __m128i temp128b;
  //
	// __m64 test_plus_128;
	// __m64 test_minus_128;



	for(int i=0; i<16; i++){
		temp_128 = _mm_load_si128(&in_vec[i]); //load 16 uint8_t elements



		//////// 1st Step:

		__m128i even_128 = _mm_shuffle_epi8(temp_128, even_mask_128);
		__m128i odd_128 = _mm_shuffle_epi8(temp_128, odd_mask_128);

		//TODO : We might need to add 1 before doing the average. ..
		avg_plus_128 = _mm_avg_epu8(even_128, odd_128);
		avg_minus_128 = _mm_avg_epu8(even_128, _mm_sub_epi8(zero128, odd_128));
		avg_plus_128 = _mm_shuffle_epi8(avg_plus_128, _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80));
		// TODO: previous line can be smartly eliminated in all Steps!!

		//////// 2nd Step:
		__m128i even_64 = _mm_shuffle_epi8(avg_plus_128, even_mask_64);
		__m128i odd_64 = _mm_shuffle_epi8(avg_plus_128, odd_mask_64);

		avg_plus_64 = _mm_avg_epu8(even_64, odd_64);
		avg_minus_64 = _mm_avg_epu8(even_64, _mm_sub_epi8(zero128, odd_64));
		avg_plus_64 = _mm_shuffle_epi8(avg_plus_64, _mm_setr_epi8(4, 5, 6, 7, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));

		//////// 3rd Step:
		__m128i even_32 = _mm_shuffle_epi8(avg_plus_64, even_mask_32);
		__m128i odd_32 = _mm_shuffle_epi8(avg_plus_64, odd_mask_32);

		avg_plus_32 = _mm_avg_epu8(even_32, odd_32);
		avg_minus_32 = _mm_avg_epu8(even_32, _mm_sub_epi8(zero128, odd_32));
		avg_plus_32 = _mm_shuffle_epi8(avg_plus_32, _mm_setr_epi8(2, 3, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));

		//////// 4rd Step:
		__m128i even_16 = _mm_shuffle_epi8(avg_plus_32, even_mask_16);
		__m128i odd_16 = _mm_shuffle_epi8(avg_plus_32, odd_mask_16);

		avg_plus_16 = _mm_avg_epu8(even_16, odd_16);
		avg_minus_16 = _mm_avg_epu8(even_16, _mm_sub_epi8(zero128, odd_16));
		avg_plus_16 = _mm_shuffle_epi8(avg_plus_16, _mm_setr_epi8(1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));

		// adding the necessary :
		temp_128 = _mm_add_epi8(avg_minus_128, avg_minus_64);
		temp_128 = _mm_add_epi8(temp_128, avg_minus_32);
		temp_128 = _mm_add_epi8(temp_128, avg_minus_16);
		temp_128 = _mm_add_epi8(temp_128, avg_plus_16);

		// Storing back the results
		 _mm_store_si128(&out_vec[i], temp_128);

	}

		////////// TEST ROUTINE : PRINT even and odd positions ////////////////////

		// printf("Input Matrix :\n");
		// print_matrix(in_vec);
    //
		// printf("Output Matrix :\n");
		// print_matrix(output);

		// printf("Odd Matrix :\n");
		// print_matrix(test_odd_128);
		//////////////////////////////////////////////////////////////////////////


}

static inline void haar_y_simd(uint8_t *output, const uint8_t *input) {
	// TODO Vectorize me
	haar_y_scalar(output, input);

	/////// Test ///////////////////////
	//haar_y_scalar(output, input);
	////////////////////////////////////

}

NO_INLINE static void haar_simd(uint8_t *output, const uint8_t *input) {
	uint8_t tmp[ROWS*COLS] ALIGNED16;

	//////////////TEST //////////////////////
	// haar_x_simd(tmp, input);
	// haar_y_simd(output, tmp);
	/////////////////////////////////////////

	haar_x_simd(output, input);

	//TODO : uncomment the following, it's from the original code then delete previous line !!
	//haar_x_simd(tmp, input);
	//haar_y_simd(output, tmp);
}

static int64_t time_diff(struct timespec start, struct timespec end) {
    struct timespec temp;
    if (end.tv_nsec-start.tv_nsec < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
	return temp.tv_sec*1000000000+temp.tv_nsec;
}

static void benchmark(
		void (*fn)(uint8_t *, const uint8_t *),
		uint8_t *output, const uint8_t *input, size_t iterations, const char *msg) {
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    for (size_t i = 0; i < iterations; i++) {
        fn(output, input);
		}

    clock_gettime(CLOCK_REALTIME, &end);
    double avg = (double) time_diff(start, end) / iterations;
    printf("%10s:\t %.3f ns\n", msg, avg);
}

static uint8_t *alloc_matrix() {
	return memalign(16, ROWS * COLS);
}

static void init_matrix(uint8_t *matrix) {
	for (size_t y = 0; y < ROWS; y++) {
		for (size_t x = 0; x < COLS; x++) {
			matrix[y * COLS + x] = (uint8_t) (rand() & UINT8_MAX);
		}
	}
}

static bool compare_matrix(uint8_t *expected, uint8_t *actual) {
	bool correct = true;
	for (size_t y = 0; y < ROWS; y++) {
		for (size_t x = 0; x < COLS; x++) {
			uint8_t e = expected[y * COLS + x];
			uint8_t a = actual[y * COLS + x];
			if (e != a) {
				printf(
					"Failed at (y=%zu, x=%zu): expected=%u, actual=%u\n",
					y, x, e, a
				);
				correct = false;
			}
		}
	}
	return correct;
}

///////////// To comment out later ///////////////
 void print_matrix(uint8_t *matrix){
	for(size_t y=0; y<ROWS; y++){
		for(size_t x=0; x<COLS; x++){
			printf("%3d  ", matrix[y*COLS + x]);
		}
		printf("\n");
	}
}
////////////////////////////////////////////////

int main() {
	uint8_t *input = alloc_matrix();
	uint8_t *output_scalar = alloc_matrix();
	uint8_t *output_simd = alloc_matrix();

	/* Check for correctness */
	for (size_t n = 0; n < 100; n++) {
		init_matrix(input);
		///////////////To comment out later ////////////////
		printf("Init matrix :\n");
		print_matrix(input);
		///////////////////////////////////////////////////
		haar_scalar(output_scalar, input);
		///////////////comment out later /////////////////
		printf("output scalar matrix :\n");
		print_matrix(output_scalar);
		/////////////////////////////////////////////////
		haar_simd(output_simd, input);
		///////////////comment out later /////////////////
		printf("output simd_x matrix :\n");
		print_matrix(output_simd);
		/////////////////////////////////////////////////


		if (!compare_matrix(output_scalar, output_simd)) {
			break;
		}
		/////////////////////////////////////////////////
		printf("n = %lu\n", n); //To comment out later
		////////////////////////////////////////////////
	}

	/* Benchmark */
	init_matrix(input);
	benchmark(haar_scalar, output_scalar, input, 3000000, "scalar");
	benchmark(haar_simd, output_simd, input, 3000000, "simd");
	benchmark(haar_x_scalar, output_scalar, input, 3000000, "scalar_x");
	benchmark(haar_x_simd, output_simd, input, 3000000, "simd_x");
	benchmark(haar_y_scalar, output_scalar, input, 3000000, "scalar_y");
	benchmark(haar_y_simd, output_simd, input, 3000000, "simd_y");

	return 1;
}
