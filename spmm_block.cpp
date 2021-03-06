/*Copyright (c) [2021] [Jose Nunez-Yanez (eejlny@bristol.ac.uk)]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
* This file has been written at University of Bristol
* for the HOPWARE/MINET project
*
* 
* author    : Jose Nunez-Yanez eejlny@bristol.ac.uk
* date      : 1 October 2021
*/

#include <hls_stream.h>
#include "spmm_block.h"

void spmm_kernel(
		ap_uint<2> ternary,
		u32 *rowSize_local_rs,
		u32 *rowSize_local_nrs,
		u32 *columnIndex,
		DATA_TYPE *values,
		DATA_TYPE_OUT *y,
		DATA_TYPE *x_local,
		u32 row_size,
		u32 nnz,
		u32 new_nnz
) {

	#pragma HLS DATAFLOW
	u32 row_size_tmp=0;
	u32 j = 0;

	DATA_TYPE_OUT y_tmp = 0;
	u32 row_counter = 0;
	//u32 index_counter = 0;
	//u32 y_row = 0;

	u32 row_size_remains = 0;

	for (u32 i = 0; i < new_nnz; i+=II) {
		#pragma HLS pipeline
		if (row_size_tmp == 0) {
			row_size_tmp = rowSize_local_nrs[j];
			row_size_remains = 0;
			y_tmp = 0;
			row_counter = rowSize_local_rs[j++];
		}

		DATA_TYPE_OUT y_local = 0;

		for (u32 p = 0; p < II; p++) {

			row_size_remains++;
			if (row_size_remains > row_counter) {
				y_local +=  0;
			} else {
				//DATA_TYPE v = values[index_counter];
				//u32 ci = columnIndex[index_counter++];
				 if(ternary == 0)
				 {
					for(int z = 0; z < DTYPE_LENGTH; z+=8) {
							#pragma HLS UNROLL
							ap_int<8> v_val = v.range(z+7,z);
							ap_int<8> x_temp = x_local[ci].range(z+7,z);
							ap_int<8> C_val;
							C_val = v_val*x_temp;
							y_local += C_val;
					}
				 }
				 else if (ternary == 1)
				 {
					for(int z = 0; z < DTYPE_LENGTH; z+=2) {
							#pragma HLS UNROLL
							ap_int<2> v_val = v.range(z+1,z);
							ap_int<2> x_temp = x_local[ci].range(z+1,z);
							ap_int<2> C_val;
							C_val = v_val*x_temp;
							y_local += C_val;
					}
				 }
				 else
				 {
					for(int z = 0; z < DTYPE_LENGTH; z+=4) {
							#pragma HLS UNROLL
							ap_int<4> v_val = v.range(z+3,z);
							ap_int<4> x_temp = x_local[ci].range(z+3,z);
							ap_int<4> C_val;
							C_val = v_val*x_temp;
							y_local += C_val;
					}
				 }
			}
		} //p loop

		y_tmp += y_local;
		row_size_tmp -= II;

		if (row_size_tmp == 0) {
			//y_fifo.write(y_tmp);
			y[y_row] = y_tmp;
			y_row += 1;
		}
	}
}

void spmm(
		ap_uint<2> ternary,
		u32 *rowPtr,

		u32 *columnIndex_0,
		u32 *columnIndex_1,
		u32 *columnIndex_2,
		u32 *columnIndex_3,

		DATA_TYPE *values_0,
		DATA_TYPE *values_1,
		DATA_TYPE *values_2,
		DATA_TYPE *values_3,

		DATA_TYPE_OUT *y_0,
		DATA_TYPE_OUT *y_1,
		DATA_TYPE_OUT *y_2,
		DATA_TYPE_OUT *y_3,

		DATA_TYPE_X *x,
		u32 no_vectors,

		u32 col_size,
		u32 row_size,
		u32 nnz,

		u32 begin,
		u32 end,

		u32 first_rowPrt_value

		) {
	//#pragma HLS DATAFLOW

	u32 rowSizeNew_local_rs[NO_HW_THREAD][ROW_SIZE_THREAD_MAX];
	u32 rowSizeNew_local_nrs[NO_HW_THREAD][ROW_SIZE_THREAD_MAX];

    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_rs complete dim=1
    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_rs cyclic  factor=4 dim=2

    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_nrs complete dim=1
    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_nrs cyclic factor=4 dim=2

	//================================================

	DATA_TYPE x_local[NO_HW_THREAD][COL_SIZE_MAX];
	#pragma HLS ARRAY_PARTITION variable=x_local complete dim=1
	#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4 dim=2
	//================================================

	u32 row_size_threads[NO_HW_THREAD];
	u32 values_offset_threads[NO_HW_THREAD];
	u32 row_offset_threads[NO_HW_THREAD];
	u32 nnz_threads[NO_HW_THREAD];
	u32 new_nnz_threads[NO_HW_THREAD];
	#pragma HLS ARRAY_PARTITION variable=row_size_threads complete dim=1
	#pragma HLS ARRAY_PARTITION variable=values_offset_threads complete dim=1
	#pragma HLS ARRAY_PARTITION variable=row_offset_threads complete dim=1
	#pragma HLS ARRAY_PARTITION variable=nnz_threads complete dim=1
	#pragma HLS ARRAY_PARTITION variable=new_nnz_threads complete dim=1
	
	u32 local_no_vectors = no_vectors;
	u32 local_col_size = col_size;


	//=======================================================

	{

		u32 ideal_nnz = nnz / NO_HW_THREAD;
		u32 local_begin = begin;
		u32 local_end = end;
		u32* local_rowPtr = rowPtr;
		int local_II = II;
		

		for (u32 i = 0; i < NO_HW_THREAD; i++) {
			#pragma HLS UNROLL
			row_size_threads[i] = 0;
			nnz_threads[i] = 0;
			new_nnz_threads[i] = 0;
		}

		u32 nrs = 0;
		u32 new_nnz = 0;
		u32 j = 0;
		u32 prev_index = first_rowPrt_value;
		u32 k = 0;

		for (u32 i = 0; i < local_end-local_begin; i++) {
			#pragma HLS PIPELINE
			u32 current_index= local_rowPtr[i+local_begin+1];
			u32 rs = (current_index - prev_index);

			if (rs == 0) {
				nrs = local_II;
				new_nnz = local_II;
			} else if (rs%local_II == 0) {
				nrs = rs;
				new_nnz = 0;
			} else {
				nrs = rs + (local_II-rs%local_II);
				new_nnz = (local_II-rs%local_II);
			}

			u32 t = nnz_threads[j] + rs;
			prev_index = current_index;

			if (t < ideal_nnz) {
				nnz_threads[j] = t;
			} else {
				if (j+1 < NO_HW_THREAD) {

					j++;
					k=0;
					nnz_threads[j] = rs;

				} else {
					nnz_threads[j] = t;
				}
			}
			row_size_threads[j]++;
			new_nnz_threads[j] += new_nnz;
			rowSizeNew_local_rs[j][k]  = rs;
			rowSizeNew_local_nrs[j][k] = nrs;
			k++;
		}


		for (u32 i = 0; i < NO_HW_THREAD; i++) {
			#pragma HLS UNROLL
			new_nnz_threads[i] += nnz_threads[i];
		}

		values_offset_threads[0] = 0;
		row_offset_threads[0] = 0;

		for (u32 i = 1; i < NO_HW_THREAD; i++) {
			#pragma HLS UNROLL
			values_offset_threads[i] = values_offset_threads[i-1]+nnz_threads[i-1];
			row_offset_threads[i] = row_offset_threads[i-1] + row_size_threads[i-1];
		}
	}

	//for (u32 i=0; i<(col_size); i++){
	//#pragma HLS pipeline
	//		for (u32 j=0; j<(NO_HW_THREAD); j++){
	//			x_local[j][i] = x[i];
	//		}
	//	}


//=======================================================================================

	for (u32 nv=0; nv < local_no_vectors; nv++){

		//for (u32 i=0; i<col_size; i+=4){
		//#pragma HLS pipeline
		//		for (u32 j=0; j<(NO_HW_THREAD); j++){
		//			DATA_TYPE_X x_wide = x[nv*(col_size>>2) + (i>>2)];
		//			x_local[j][i] = x_wide.range(7,0);
		//			x_local[j][i+1] = x_wide.range(15,8);
		//			x_local[j][i+2] = x_wide.range(23,16);
		//			x_local[j][i+3] = x_wide.range(31,24);
		//		}
		//	}


		//if(!ternary)
		//{
			for (u32 i=0; i<(local_col_size>>2); i++){
				#pragma HLS pipeline
				for (u32 j=0; j<(NO_HW_THREAD); j++){
					DATA_TYPE_X x_wide = x[nv*(local_col_size>>2) + i];
					x_local[j][i*4] = x_wide.range(7,0);
					x_local[j][i*4+1] = x_wide.range(15,8);
					x_local[j][i*4+2] = x_wide.range(23,16);
					x_local[j][i*4+3] = x_wide.range(31,24);
				}
			}
		//}
		//else
		//{
		//	for (u32 i=0; i<(col_size>>2); i++){
		//		#pragma HLS pipeline
		//				for (u32 j=0; j<(NO_HW_THREAD); j++){
		//					DATA_TYPE_X x_wide = x[nv*(col_size>>2) + i];
		//					x_local[j][i*4] = x_wide.range(7,0);
		//					x_local[j][i*4+1] = x_wide.range(15,8);
		//					x_local[j][i*4+2] = x_wide.range(23,16);
		//					x_local[j][i*4+3] = x_wide.range(31,24);
		//				}
		//			}

		//}



//=======================================================================================
			
			//u32 i;
			for (int i = 0; i < NO_HW_THREAD; i++) {
			//i = 0;
				#pragma HLS pipeline
			//#pragma HLS DATAFLOW
				spmm_kernel(
						ternary,
						rowSizeNew_local_rs[i],
						rowSizeNew_local_nrs[i],
						columnIndex_0 + first_rowPrt_value + values_offset_threads[i],
						values_0 + first_rowPrt_value + values_offset_threads[i],
						y_0 + begin + nv*row_size + row_offset_threads[i],
						x_local[i],
						row_size_threads[i],
						nnz_threads[i],
						new_nnz_threads[i]
				);
			}
			/*
			//i = 1;
			spmm_kernel(
					ternary,
					rowSizeNew_local_rs[1],
					rowSizeNew_local_nrs[1],
					columnIndex_1 + first_rowPrt_value + values_offset_threads[1],
					values_1 + first_rowPrt_value + values_offset_threads[1],
					y_1 + begin + nv*row_size + row_offset_threads[1],
					x_local[1],
					row_size_threads[1],
					nnz_threads[1],
					new_nnz_threads[1]
			);

			//i = 2;
			spmm_kernel(
					ternary,
					rowSizeNew_local_rs[2],
					rowSizeNew_local_nrs[2],
					columnIndex_2 + first_rowPrt_value + values_offset_threads[2],
					values_2 + first_rowPrt_value + values_offset_threads[2],
					y_2 + begin + nv*row_size + row_offset_threads[2],
					x_local[2],
					row_size_threads[2],
					nnz_threads[2],
					new_nnz_threads[2]
			);

			//i = 3;
			spmm_kernel(
					ternary,
					rowSizeNew_local_rs[3],
					rowSizeNew_local_nrs[3],
					columnIndex_3 + first_rowPrt_value + values_offset_threads[3],
					values_3 + first_rowPrt_value + values_offset_threads[3],
					y_3 + begin + nv*row_size + row_offset_threads[3],
					x_local[3],
					row_size_threads[3],
					nnz_threads[3],
					new_nnz_threads[3]
			);
			*/
			
		}
}

void spmm_block(
		ap_uint<2> ternary,
		DATA_TYPE *values,
		u32       *colIndices,
		u32       *rowPtr,
		DATA_TYPE_X *x,
		u32        no_vectors,

		DATA_TYPE_OUT *y,
		u32        rows,
		u32        cols,
		u32        nnz,
		u32        begin,
		u32        end

		) {

    //#pragma SDS resource(1)
	spmm(
			ternary,
			rowPtr,

			colIndices,
			colIndices,
			colIndices,
			colIndices,

			values,
			values,
			values,
			values,

			y,
			y,
			y,
			y,

			x,
			no_vectors,

			cols,
			rows,
			nnz,

			begin,
			end,
			rowPtr[begin]
			);
}
