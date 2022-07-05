#include <iostream>
#include <fstream>
#include <string.h>
//#include <sys/time.h>
#include <algorithm>
#include <vector>

#include "xcl2.hpp"
#include <CL/cl.h>
#include <CL/cl2.hpp>
#include "math.h"
//#include <hls_stream.h>
#include "spmm_block.h"

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

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
	
	/*
	hls::stream<DATA_TYPE>       values_fifo;
	#pragma HLS STREAM variable=values_fifo depth=4 dim=1
	hls::stream<u32>             col_indices_fifo;
	#pragma HLS STREAM variable=col_indices_fifo depth=4 dim=1
	hls::stream<DATA_TYPE_OUT>       y_fifo;
	#pragma HLS STREAM variable=y_fifo depth=4 dim=1
	*/
	DATA_TYPE *values_fifo = values;
	u32 *col_indices_fifo = columnIndex;
	DATA_TYPE_OUT y_fifo;
	/*
	for (u32 i = 0; i < nnz; i+=1) {
		#pragma HLS pipeline
		values_fifo << values[i];
		col_indices_fifo << columnIndex[i];
	}
	*/

	u32 row_size_remains = 0;

	for (u32 i = 0; i < new_nnz; i+=II) {
		#pragma HLS pipeline
		if (row_size_tmp == 0) {
			row_size_tmp = rowSize_local_nrs[j];
			row_size_remains = 0;
			y_tmp = 0;
			row_counter	= rowSize_local_rs[j++];
		}

		DATA_TYPE_OUT y_local = 0;

		for (u32 p = 0; p < II; p++) {
			row_size_remains++;
			if (row_size_remains > row_counter) {
				y_local +=  0;
			} else {
				DATA_TYPE v = *values_fifo;
				u32 ci = *col_indices_fifo;
				//y_local +=  v*x_local[ci];
				 if(ternary == 0)
				 {
					for(int z = 0; z < DTYPE_LENGTH; z+=8) {
							ap_int<8> v_val = v.range(z+7,z);
							ap_int<8> x_temp = x_local[ci].range(z+7,z);
							//y_local +=  v_val*x_local[ci].range(z+7,z);
							ap_int<8> C_val;
							C_val = v_val*x_temp;
							y_local += C_val;
							//std::cout << "y_local  " << y_local << std::endl;
					}
				 }
				 else if (ternary == 1)
				 {
					for(int z = 0; z < DTYPE_LENGTH; z+=2) {

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
		//std::cout << "y_tmp  " << y_tmp << std::endl;
		/*
		if (row_size_tmp == 0) {
			y_fifo << y_tmp;

		}
		*/
		if (row_size_tmp == 0) {
			y_fifo == y_tmp;
		}
	}

	for (u32 i = 0; i < row_size; i+=1) {
		#pragma HLS pipeline
		y[i] = y_fifo.read();
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
		u32        no_vectors,

		u32 col_size,
		u32 row_size,
		u32 nnz,

		u32 begin,
		u32 end,

		u32 first_rowPrt_value

		) {
	#pragma HLS DATAFLOW

	u32 rowSizeNew_local_rs[NO_HW_THREAD][ROW_SIZE_THREAD_MAX];
	u32 rowSizeNew_local_nrs[NO_HW_THREAD][ROW_SIZE_THREAD_MAX];

    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_rs complete dim=1
    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_rs cyclic  factor=4 dim=2

    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_nrs complete dim=1
    #pragma HLS ARRAY_PARTITION variable=rowSizeNew_local_nrs cyclic factor=4 dim=2

	//================================================

	DATA_TYPE x_local[NO_HW_THREAD][COL_SIZE_MAX];
	#pragma HLS ARRAY_PARTITION variable=x_local complete dim=1
	//#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4 dim=2
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


	//=======================================================

	{

		u32 ideal_nnz = nnz / NO_HW_THREAD;

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

		for (u32 i = 0; i < end-begin; i++) {
			#pragma HLS PIPELINE
			u32 current_index= rowPtr[i+begin+1];
			u32 rs = (current_index - prev_index);

			if (rs == 0) {
				nrs = II;
				new_nnz = II;
			} else if (rs%II == 0) {
				nrs = rs;
				new_nnz = 0;
			} else {
				nrs = rs + (II-rs%II);
				new_nnz = (II-rs%II);
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

	for (u32 nv=0; nv < no_vectors; nv++){

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
			for (u32 i=0; i<(col_size>>2); i++){
				#pragma HLS pipeline
				for (u32 j=0; j<(NO_HW_THREAD); j++){
					DATA_TYPE_X x_wide = x[nv*(col_size>>2) + i];
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

			u32 i;

			i = 0;
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

			i = 1;
			spmm_kernel(
					ternary,
					rowSizeNew_local_rs[i],
					rowSizeNew_local_nrs[i],
					columnIndex_1 + first_rowPrt_value + values_offset_threads[i],
					values_1 + first_rowPrt_value + values_offset_threads[i],
					y_1 + begin + nv*row_size + row_offset_threads[i],
					x_local[i],
					row_size_threads[i],
					nnz_threads[i],
					new_nnz_threads[i]
			);

			i = 2;
			spmm_kernel(
					ternary,
					rowSizeNew_local_rs[i],
					rowSizeNew_local_nrs[i],
					columnIndex_2 + first_rowPrt_value + values_offset_threads[i],
					values_2 + first_rowPrt_value + values_offset_threads[i],
					y_2 + begin + nv*row_size + row_offset_threads[i],
					x_local[i],
					row_size_threads[i],
					nnz_threads[i],
					new_nnz_threads[i]
			);

			i = 3;
			spmm_kernel(
					ternary,
					rowSizeNew_local_rs[i],
					rowSizeNew_local_nrs[i],
					columnIndex_3 + first_rowPrt_value + values_offset_threads[i],
					values_3 + first_rowPrt_value + values_offset_threads[i],
					y_3 + begin + nv*row_size + row_offset_threads[i],
					x_local[i],
					row_size_threads[i],
					nnz_threads[i],
					new_nnz_threads[i]
			);
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
/*
double getTimestamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_usec + tv.tv_sec * 1e6;
}
void msleep(u32 c) {
	usleep(c * 1000);
}

float cpu_percentage;
float low_percentage;
float high_percentage;
*/

/*
u32 read_mtx_spmm(u32* row_size, u32* col_size, u32* nnz,FILE *fp_input) {

    std::cout << "read_csr: check point 1" << std::endl;

	u32 r;
	u32 c;
	DATA_TYPE v;

	u32 nnzeo_false = 0;
	if (fp_input != NULL) {
        std::cout << "read_mtx_spmm: check point 2" << std::endl;
		char line[1000];
		//while (fgets(line, sizeof line, fp_input) != NULL) {// read a line from a file
        while (fgets(line, sizeof(line), fp_input) != NULL) {
			if (line[0] != '%') {
				sscanf(line, "%d %d %d", row_size, col_size, nnz);
				std::cout << "row_size = " <<  *row_size << " col_size = " << *col_size << " nnz = " << *nnz << std::endl;
                std::cout << "read_mtx_spmm: check point 3" << std::endl;

				{
					array_values = (DATA_TYPE*)fpga_malloc(*nnz * sizeof(DATA_TYPE));
					if (!array_values) {
						std::cout << "unable to allocate memory array_values" << std::endl;
						exit(1);
					}

					array_colIndices = (u32*)fpga_malloc(*nnz * sizeof(u32));
					if (!array_colIndices) {
						std::cout << "unable to allocate memory array_colIndices" << std::endl;
						exit(1);
					}

					array_rowPtr = (u32*)fpga_malloc((*row_size + 1) * sizeof(u32));
					if (!array_rowPtr) {
						std::cout << "unable to allocate memory array_rowPtr" << std::endl;
						exit(1);
					}

					array_x = (DATA_TYPE_X*)fpga_malloc(*col_size * no_vectors * sizeof(DATA_TYPE_X)/4);
					if (!array_x) {
						std::cout << "unable to allocate memory array_x" << std::endl;
						exit(1);
					}

					array_y = (DATA_TYPE_OUT*)fpga_malloc(*row_size * no_vectors * sizeof(DATA_TYPE_OUT));
					if (!array_y) {
						std::cout << "unable to allocate memory array_y" << std::endl;
						exit(1);
					}

					array_y_golden = (DATA_TYPE_OUT*)malloc(*row_size * no_vectors * sizeof(DATA_TYPE_OUT));
					if (!array_y_golden) {
						std::cout << "unable to allocate memory array_y_golden" << std::endl;
						exit(1);
					}
				}

				u32 line_number = 0;
				//while (fgets(line, sizeof line, fp_input) != NULL) {// read a line from a file
                while (fgets(line, sizeof(line), fp_input) != NULL) {
					if (line_number < *nnz) {

						sscanf(line, "%d %d", &c, &v);

						//printf("colindices %d val %f\n", c, v);
						//std::cout << "colindices" << c << " val " << v << std::endl;

						*(array_colIndices + line_number) = c;

						*(array_values + line_number) = v;

					}
					else {
						sscanf(line, "%d", &r);

						//printf("rowptr %d \n", r);
						//std::cout << "rowptr " << c << std::endl;

						*(array_rowPtr + (line_number - (*nnz))) = r;
					}
					line_number++;
				}
			}
		}
	}
	else {
		//perror(argv[1]); //print the error message on stderr.
		std::cout << "Error with input file name" << std::endl;
		exit(1);
	}

	return 0;
}
*/

u32 golden_spmm_ternary(DATA_TYPE * values, u32 *row_ptr, u32* col_indices, DATA_TYPE_X * x, u32 no_vectors, DATA_TYPE_OUT *y, u32 row_size, u32 col_size) {

    std::cout << "gold_spmm_ternary: check point 1" << std::endl;
	u32 nvc = 0, i = 0, j = 0, rowStart = 0, rowEnd = row_size;

	DATA_TYPE_OUT y0 = 0;
	u32 last_j = 0;
	for (nvc = 0; nvc < no_vectors; nvc++) {
		for (i = rowStart; i < rowEnd; ++i) {
			y0 = 0;
			for (j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
				//y0 += values[j] * x[nvc*col_size+col_indices[j]];
				for(int z = 0; z < DTYPE_LENGTH; z+=2) {
					            DATA_TYPE values_val1 = values[j];
					        	ap_int<2> values_val = values_val1.range(z+1,z);
					        	int x_value = nvc*col_size+col_indices[j];
					        	int x_up = x_value >> 4;
					        	int x_down = (x_value & 0xF);
								DATA_TYPE values_val_temp = values_val;
						       	ap_int<2> x_temp;
								switch(values_val_temp)
								{
									 case 0:
										 //std::cout << "C is" << C[j] << std::endl;
										 break;
									 case 1:
										 x_temp = x[x_up].range(x_down*2+1,x_down*2);
										 y0 += x_temp;
										 //std::cout << "B is" << b[k][j].range(z+1,z) << std::endl;
										 //std::cout << "C is" << C[j] << std::endl;
										 break;
									 case -1:
										 x_temp = x[x_up].range(x_down*2+1,x_down*2);
										 y0 -= x_temp;
										 //std::cout << "B is" << b[k][j].range(z+1,z) << std::endl;
										 //std::cout << "C is" << C[j] << std::endl;
										 break;
								}

				}
				//std::cout << "y0 is " << y0 << std::endl;
			}
			y[nvc*row_size+i] = y0;
		}
	}

	return 0;
}

u32 golden_spmm_byte(DATA_TYPE * values, u32 *row_ptr, u32* col_indices, DATA_TYPE_X * x, u32 no_vectors, DATA_TYPE_OUT *y, u32 row_size, u32 col_size) {

    std::cout << "golden_spmm_byte: check point 2" << std::endl;
	u32 nvc = 0, i = 0, j = 0, rowStart = 0, rowEnd = row_size;

	DATA_TYPE_OUT y0 = 0;
	u32 last_j = 0;
	for (nvc = 0; nvc < no_vectors; nvc++) {
		for (i = rowStart; i < rowEnd; ++i) {
			y0 = 0;
			for (j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
				//y0 += values[j] * x[nvc*col_size+col_indices[j]];
				for(int z = 0; z < DTYPE_LENGTH; z+=8) {
					            DATA_TYPE values_val1 = values[j];
								ap_int<8> values_val = values_val1.range(z+7,z);
								int x_value = nvc*col_size+col_indices[j];
								int x_up = x_value >> 2;
								int x_down = (x_value & 0x3);
						       	y0 += values_val * x[x_up].range(x_down*8+7,x_down*8);
						       	//std::cout << "y0 " << y0 << std::endl;

				}
				//std::cout << "y0 is " << y0 << std::endl;
			}
			//std::cout << "y0 is " << y0 << std::endl;
			y[nvc*row_size+i] = y0;
		}
	}

	return 0;
}

u32 golden_spmm_quad(DATA_TYPE * values, u32 *row_ptr, u32* col_indices, DATA_TYPE_X * x, u32 no_vectors, DATA_TYPE_OUT *y, u32 row_size, u32 col_size) {

    std::cout << "golden_spmm_quad: check point 3" << std::endl;
	u32 nvc = 0, i = 0, j = 0, rowStart = 0, rowEnd = row_size;

	DATA_TYPE_OUT y0 = 0;
	u32 last_j = 0;
	for (nvc = 0; nvc < no_vectors; nvc++) {
		for (i = rowStart; i < rowEnd; ++i) {
			y0 = 0;
			for (j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
				//y0 += values[j] * x[nvc*col_size+col_indices[j]];
				for(int z = 0; z < DTYPE_LENGTH; z+=4) {
					            DATA_TYPE values_val1 = values[j];
								ap_int<4> values_val = values_val1.range(z+3,z);
								int x_value = nvc*col_size+col_indices[j];
								int x_up = x_value >> 3;
								int x_down = (x_value & 0x7);
						       	y0 += values_val * x[x_up].range(x_down*4+3,x_down*4);
						       	//std::cout << "y0 " << y0 << std::endl;

				}
				//std::cout << "y0 is " << y0 << std::endl;
			}
			//std::cout << "y0 is " << y0 << std::endl;
			y[nvc*row_size+i] = y0;
		}
	}

	return 0;
}

void init_array(ap_uint<2> ternary, DATA_TYPE_X *x, u32 row, u32 col)
{
    if(ternary==0)
	{
		for (u32 i = 0; i < row; i++) {
			for (u32 j = 0; j < (col>>2); j++) {
				x[i*(col>>2)+j] = 0x01010101;
			}
		}
	}
	else if (ternary==1)
	{
		for (u32 i = 0; i < row; i++) {
			for (u32 j = 0; j < (col>>2); j++) {
				x[i*(col>>2)+j] = 0x55555555;
			}
		}
	}
	else
	{
		for (u32 i = 0; i < row; i++) {
			for (u32 j = 0; j < (col>>2); j++) {
				x[i*(col>>2)+j] = 0x11111111;
			}
		}
	}
}

static int result_check(DATA_TYPE_OUT *y, DATA_TYPE_OUT *y_golden, u32 row, u32 col)
{
	for (int i = 0; i < row * col; i++) {
		if (y_golden[i] != y[i]) {
			std::cout 	<< "Mismatch: data index= " << i << " golden = " << y_golden[i]
						<< ", kernel = " << y[i] << std::endl;
			return 1;
		}
	}
    std::cout 	<< "TEST PASSED !" <<  std::endl;
	return 0;
}

//MAIN
int main(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << " myFile" << " ternary" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];
    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        program = cl::Program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl = cl::Kernel(program, "spmm_block", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

	FILE *fp_input;
	fp_input = fopen(argv[2], "r");
    //fp_input = argv[2];
    ap_uint<2> S_ternary = atoi(argv[3]);

	u32 r;
	u32 c;
	DATA_TYPE v;

    DATA_TYPE *array_values;
    u32* array_colIndices;
    u32* array_rowPtr;

    u32 row_size;
    u32 col_size;
    u32 nnz;

	if (fp_input != NULL) {
        std::cout << "read_mtx_spmm: check point 2" << std::endl;
		char line[1000];
	std::cout << "has defined a char line[1000]" << std::endl;
        while (fgets(line, sizeof(line), fp_input) != NULL) {
		std::cout << "has entered while" << std::endl;
			if (line[0] != '%') {
				std::cout << "has entered if, start to sscanf" << std::endl;
				sscanf(line, "%u %u %u", &row_size, &col_size, &nnz);
                //std::cout << "row_size = " <<  *row_size << " col_size = " << *col_size << " nnz = " << *nnz << std::endl;
                std::cout << "row_size = " <<  row_size << " col_size = " << col_size << " nnz = " << nnz << std::endl;
                std::cout << "read_mtx_spmm: check point 3" << std::endl;
				
				array_values = new DATA_TYPE[nnz];
				array_colIndices = new u32[nnz];
				array_rowPtr = new u32[row_size + 1];
				
				u32 line_number = 0;
                while (fgets(line, sizeof(line), fp_input) != NULL) {
					if (line_number < nnz) {
						std::cout << "has entered if, start to sscanf" << std::endl;
						sscanf(line, "%d %d", &c, &v);

						//printf("colindices %d val %f\n", c, v);
						//std::cout << "colindices" << c << " val " << v << std::endl;

						*(array_colIndices + line_number) = c;
						std::cout << "array_colIndices = " << *array_colIndices << std::endl;
						*(array_values + line_number) = v;
						std::cout << "array_values = " << *array_values << std::endl;
						std::cout << "(if) Pass 'something could go wrong' stage" << std::endl;

					}
					else {
						sscanf(line, "%d", &r);

						//printf("rowptr %d \n", r);
						//std::cout << "rowptr " << c << std::endl;

						*(array_rowPtr + (line_number - (nnz))) = r;
						std::cout << "(else) Pass 'something could go wrong' stage" << std::endl;
					}
					line_number++;
				}
			}
		}
	}
	else {
		//perror(argv[1]); //print the error message on stderr.
		std::cout << "Error with input file name" << std::endl;
		exit(1);
	}

    u32 no_vectors = 512;

    // Map our user-allocated buffers as OpenCL buffers using a shared host pointer
    OCL_CHECK(err, cl::Buffer buffer_array_values(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , nnz * sizeof(DATA_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_array_colIndices(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , nnz * sizeof(u32), NULL, &err));    
    OCL_CHECK(err, cl::Buffer buffer_array_rowPtr(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , (row_size + 1) * sizeof(u32), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_array_x(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , col_size * no_vectors * sizeof(DATA_TYPE_X)/4, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_array_y(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR , row_size * no_vectors * sizeof(DATA_TYPE_OUT), NULL, &err));

	// For buffer buffer_array_y_golden, since we aren't using it for a kernel we'll specify the bank allocation
    cl_mem_ext_ptr_t bank_ext;
    bank_ext.flags = 0 | XCL_MEM_TOPOLOGY;
    bank_ext.obj   = NULL;
    bank_ext.param = 0;
	OCL_CHECK(err, cl::Buffer buffer_array_y_golden(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_EXT_PTR_XILINX, row_size * no_vectors * sizeof(DATA_TYPE_OUT), &bank_ext, &err));

    DATA_TYPE_X *array_x;
    DATA_TYPE_OUT *array_y;
    DATA_TYPE_OUT * array_y_golden;

    u32 S_begin = 0;
    u32 S_end = row_size;

    // Set the kernal argument
	/*
    int narg = 0;
	OCL_CHECK(err, err = krnl.setArg(narg++, S_ternary));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_values));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_colIndices));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_rowPtr));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_x));
    OCL_CHECK(err, err = krnl.setArg(narg++, no_vectors));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_y));
    OCL_CHECK(err, err = krnl.setArg(narg++, row_size));
    OCL_CHECK(err, err = krnl.setArg(narg++, col_size));
    OCL_CHECK(err, err = krnl.setArg(narg++, nnz));
    OCL_CHECK(err, err = krnl.setArg(narg++, S_begin));
    OCL_CHECK(err, err = krnl.setArg(narg++, S_end));
	*/

    //Map buffers to userspace pointers
    OCL_CHECK(err, array_values = (DATA_TYPE*)q.enqueueMapBuffer(buffer_array_values, CL_TRUE, CL_MAP_WRITE, 0, nnz * sizeof(DATA_TYPE), nullptr, nullptr, &err));
    OCL_CHECK(err, array_colIndices = (u32*)q.enqueueMapBuffer(buffer_array_colIndices, CL_TRUE, CL_MAP_WRITE, 0, nnz * sizeof(u32), nullptr, nullptr, &err));
    OCL_CHECK(err, array_rowPtr = (u32*)q.enqueueMapBuffer(buffer_array_rowPtr, CL_TRUE, CL_MAP_WRITE, 0, (row_size + 1) * sizeof(u32), nullptr, nullptr, &err));
    OCL_CHECK(err, array_x = (DATA_TYPE_X*)q.enqueueMapBuffer(buffer_array_x, CL_TRUE, CL_MAP_WRITE, 0, col_size * no_vectors * sizeof(DATA_TYPE_X)/4, nullptr, nullptr, &err));
	OCL_CHECK(err, array_y = (DATA_TYPE_OUT*)q.enqueueMapBuffer(buffer_array_y, CL_TRUE, CL_MAP_READ, 0, row_size * no_vectors * sizeof(DATA_TYPE_OUT), nullptr, nullptr, &err));
	OCL_CHECK(err, array_y_golden = (DATA_TYPE_OUT*)q.enqueueMapBuffer(buffer_array_y_golden, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, row_size * no_vectors * sizeof(DATA_TYPE_OUT), nullptr, nullptr, &err));

    //Initialization
	std::cout << "Start to init_array " << std::endl;

    init_array(S_ternary, array_x, no_vectors, col_size);

	//double start_time, end_time, execution_time;
	std::cout << "Start to kernel : spmm_block " << std::endl;

    spmm_block(
		S_ternary,
		array_values,
		array_colIndices,
		array_rowPtr,
		array_x,
		no_vectors,
		array_y,
		row_size,
		col_size,
		nnz,
		S_begin,
		S_end
	);
    // Date will be migrate to the kernal space
	//OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_array_values, buffer_array_colIndices, buffer_array_rowPtr, buffer_array_x}, 0));
    
    // Lauch the kernal
    //OCL_CHECK(err, err = q.enqueueTask(krnl));
    
    // To view the results, this call will transfer the data from FPGA to the host

	// Rather than manually enqueueing a migration, we can instead just map the buffer. 
	// The OpenCL runtime will recognize that the buffer contents are currently resident in 
	// the Alveo Data Center accelerator card global memory and will take care of 
	// migrating the buffer back to the host for us. This is a coding style choice you must make.

    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_array_y}, CL_MIGRATE_MEM_OBJECT_HOST));
    
    //q.finish();

    /*
	start_time = getTimestamp();
	
	end_time = getTimestamp();
	execution_time = (end_time - start_time) / (1000);
	std::cout << "FPGA " << " Total execution_time = " << execution_time << " msec" << std::endl;

    std::cout << "Start to mmult_golden " << std::endl;
	start_time = getTimestamp();
    */
   
    std::cout << "Start to mmult_golden " << std::endl;

	if (S_ternary==0)
    {
        golden_spmm_byte(
            array_values,
            array_rowPtr,
            array_colIndices,
            array_x,
            no_vectors,
            array_y_golden,
            row_size,
            col_size
        );
    }
	else if (S_ternary==1)
    {
        golden_spmm_ternary(
            array_values,
            array_rowPtr,
            array_colIndices,
            array_x,
            no_vectors,
            array_y_golden,
            row_size,
            col_size
        );
    }
	else
    {
        golden_spmm_quad(
            array_values,
            array_rowPtr,
            array_colIndices,
            array_x,
            no_vectors,
            array_y_golden,
            row_size,
            col_size
        );
    }
    /*
	end_time = getTimestamp();

	execution_time = (end_time - start_time) / (1000);
	std::cout << "CPU " << " Total execution time = " << execution_time << " msec" << std::endl;
    */

    // Compare the results of the Device to the simulation
	std::cout << "Start to result_check " << std::endl;

    if(result_check(array_y, array_y_golden, row_size, no_vectors))
        return 1;

	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_values, array_values));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_colIndices, array_colIndices));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_rowPtr, array_rowPtr));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_x, array_x));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_y, array_y));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_y_golden, array_y_golden));
	q.finish();

}
