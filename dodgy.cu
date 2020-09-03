//nvcc -o flower dodgy.cu -O3 -m=64 -arch=compute_61 -code=sm_61 -Xptxas -allow-expensive-optimizations=true -Xptxas -v
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <inttypes.h>
#include <bitset>
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <mutex>
#include <queue>
#include <thread>
#include <sstream>
#include "lcg.h"
#include "util.h"
#include "gpu.h"

struct POSITION_DATA {int8_t x; int8_t y; int8_t z;};
#define POS_ARR_COUNT(array) (sizeof(array)/sizeof(POSITION_DATA))


#define BLOCK_SIZE (128LLU+64LLU)
#define WORK_SIZE (1LLU<<10)
#define WORK_INCREMENT (BLOCK_SIZE*WORK_SIZE*SEED_PROCESSING_SIZE)


//The number of circular buffer things to process
#define SEED_PROCESSING_SIZE ((1LLu<<14)*6)



__constant__ POSITION_DATA center =  { 7, 3, 7 };
__constant__ POSITION_DATA coords[] = { { 0, 0, 0 }, { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } };
__constant__ bool known[15][7][15];



__global__ __launch_bounds__(BLOCK_SIZE) void  DoGPU(uint64_t offset, uint32_t* count) {
	uint64_t seed = ((uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) + offset)*SEED_PROCESSING_SIZE;//Check
	uint64_t inital_dfz = seed&lcg::MASK;
	seed = lcg::dfz2seed(inital_dfz);
	
	
	int8_t fcount[15][7][15];
	#pragma unroll
	for(uint32_t i = 0; i<sizeof(fcount);i++)
		*(((int8_t*)fcount)+i)=0;
	
	int8_t missingcount =  POS_ARR_COUNT(coords);
	
	__shared__ POSITION_DATA posbuffer[64][BLOCK_SIZE];
	for (int time = 0; time < 64; time++) {
		uint8_t ptr = (uint8_t) (time & 63);
		uint8_t i1 = (7 + lcg::next_int<8>(seed)) - lcg::next_int<8>(seed);
		uint8_t j1 = (3 + lcg::next_int<4>(seed)) - lcg::next_int<4>(seed);
		uint8_t k1 = (7 + lcg::next_int<8>(seed)) - lcg::next_int<8>(seed);
		posbuffer[ptr][threadIdx.x] = {i1, j1, k1};
		//__prefetch_local_l1(&posbuffer[ptr+1]);
		//missingcount -= ((fcount[i1][j1][k1] - 1) >> 7) & known[i1][j1][k1];
		if (known[i1][j1][k1] && fcount[i1][j1][k1] == 0) {
			missingcount--;
		}
		fcount[i1][j1][k1]++;
	}
	
	for (uint32_t time = 0; time < (SEED_PROCESSING_SIZE / 6 + 1); time++) {
		if (missingcount == 0) {
			/*
			long DFZ = initialDFZ + 6 * time;
			if (DFZ < blockstart + blocklength) {
				System.out.println(DFZ);
			}*/
			atomicAdd(count,1);
			uint64_t DFZ = inital_dfz + 6 * time;
			printf("%llu\n",DFZ);
		}
		
		uint8_t ptr = (uint8_t) (time & 63);
		uint8_t i0 = posbuffer[ptr][threadIdx.x].x;
		uint8_t j0 = posbuffer[ptr][threadIdx.x].y;
		uint8_t k0 = posbuffer[ptr][threadIdx.x].z;
		fcount[i0][j0][k0]--;
		//missingcount += ((fcount[i0][j0][k0] - 1) >> 7) & known[i0][j0][k0];
		if (known[i0][j0][k0] && fcount[i0][j0][k0] == 0) {
			missingcount++;
		 }
		i0 = (7 + lcg::next_int<8>(seed)) - lcg::next_int<8>(seed);
		j0 = (3 + lcg::next_int<4>(seed)) - lcg::next_int<4>(seed);
		k0 = (7 + lcg::next_int<8>(seed)) - lcg::next_int<8>(seed);
		posbuffer[ptr][threadIdx.x] = {i0, j0, k0};
		//__prefetch_local_l1(&posbuffer[ptr+1]);
		//missingcount -= ((fcount[i0][j0][k0] - 1) >> 7) & known[i0][j0][k0];
		if (known[i0][j0][k0] && fcount[i0][j0][k0] == 0) {
			missingcount--;
		}
		fcount[i0][j0][k0]++;
	}
}










bool RUNNING = true;

struct GPU_BATCH_OUTPUT {uint32_t count; uint64_t* seed_buffer;};
std::mutex queueMutex;
std::queue<GPU_BATCH_OUTPUT> postProcessData;

void ProcessingWorker() {
	while(RUNNING) {
		//Wait for data
		while (true) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));//Sleep for 1000 milliseconds
			queueMutex.lock();
			if (!postProcessData.empty()){queueMutex.unlock(); break;}
			queueMutex.unlock();
		}
		//Process data
		while (true) {
			queueMutex.lock();
			if (postProcessData.empty()){queueMutex.unlock(); break;}
			GPU_BATCH_OUTPUT data = postProcessData.front();
			postProcessData.pop();
			queueMutex.unlock();
			for(uint32_t index = 0; index < data.count; index++) {
				uint64_t seed = data.seed_buffer[index];
				/*
				if (FlowerMatchCPU(seed)) {
					std::cout << "Matching seed: " << seed << std::endl;
					
					std::ofstream outfile;
					outfile.open("SEEDBOI.txt", std::ios_base::app); // append instead of overwrite
					outfile << seed << std::endl;
					outfile.close();
				}*/
			}
			
			free(data.seed_buffer);
		}
	}
}

void PostProcessing(uint32_t* counter, uint64_t* buff) {
	uint64_t* seed_buff = (uint64_t*)malloc(*counter*sizeof(uint64_t));
	GPU_ASSERT(cudaMemcpy(seed_buff, buff, *counter*sizeof(uint64_t), cudaMemcpyDeviceToHost));
	queueMutex.lock();
	postProcessData.push({*counter, seed_buff});
	queueMutex.unlock();
}








int32_t GPU_ID = 0;
int main(int argc, char* argv[]) {
	std::thread DataProcesser(ProcessingWorker);

	uint64_t START = 0;
	uint64_t END = (1LLu<<48);
	
	SETGPU(GPU_ID);
	
	uint32_t* counter;
	GPUMALLOC(counter, 1);
	
	bool known2[15][7][15];
	POSITION_DATA coordsCPU[POS_ARR_COUNT(coords)];
	POSITION_DATA centerCPU;
	
	cudaMemcpyFromSymbol(coordsCPU, coords, sizeof(coords));
	GPU_ASSERT(cudaPeekAtLastError());
	cudaMemcpyFromSymbol(&centerCPU, center, sizeof(center));
	GPU_ASSERT(cudaPeekAtLastError());
	
	
	for(uint32_t i = 0; i<sizeof(known2);i++)
		*(((bool*)known2)+i)=0;
	for(uint32_t i = 0; i<POS_ARR_COUNT(coords);i++)
		known2[coordsCPU[i].x-centerCPU.x+7][coordsCPU[i].y-centerCPU.y+3][coordsCPU[i].z-centerCPU.z+7] = 1;
	
	GPU_ASSERT(cudaMemcpyToSymbol(known, &known2, sizeof(known2)));
	GPU_ASSERT(cudaPeekAtLastError());
	
	
	
	
	
	
	
	
	
	
	
	for (uint64_t offset = START; offset < END; offset += WORK_INCREMENT) {
		//std::cout << offset << " aaaaaa\n";
		uint64_t start = millis();
		*counter = 0;
		DoGPU<<<WORK_SIZE,BLOCK_SIZE>>>(offset, counter);
		GPU_ASSERT(cudaPeekAtLastError());
		GPU_ASSERT(cudaDeviceSynchronize());
		GPU_ASSERT(cudaPeekAtLastError());
		std::cout << "Time taken: " << (millis() - start) << " speed: " << (((double)WORK_INCREMENT/(millis() - start))/1000) << " msps " <<
			" Seeds: " << *counter << " ETA: " << ((((END-offset)/WORK_INCREMENT)*(millis() - start))/1000) << " seconds" << std::endl;
	}
	RUNNING = false;
	DataProcesser.join();
}










/*
int main() {
	uint64_t start = millis();
	
	uint64_t count = 0;
	for(uint64_t seed = 0; seed < 50 000 000; seed++) 
		count += FlowerMatch(seed);
	
	std::cout << "Valid seed count: " << count << std::endl;
	std::cout << "Time in millis: " << (millis() - start) << std::endl;
	return 0;
}*/































