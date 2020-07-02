//
// Created by kranya on 03.07.2020.
//

#include <CL/opencl.h>
#include <cstdio>
#include <iostream>
#include <random>

void fill_matrix(float *ptr, size_t cnt) {
    std::random_device rd;
    std::uniform_real_distribution<float> dis(0.1, 200.0);
    for (size_t i = 0; i < cnt; ++i) {
        ptr[i] = dis(rd) / cnt;
    }
}

bool check_matrix(size_t n, size_t m, size_t k, float* a, float* b, float* c) {
    auto *res = new float[k * n]();

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t l = 0; l < k; ++l) {
                res[i * k + l] += a[i * m + j] * b[j * k + l];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t l = 0; l < k; ++l) {
            float delta = res[i * k + l] - c[i * k + l];
            float abs_delta = fabsf(delta);
            if (abs_delta >= 0.1) {
                delete[] res;
                return false;
            }
        }
    }
    delete[] res;
    return true;
}

int main() {
    cl_uint platform_number;
    clGetPlatformIDs(0, NULL, &platform_number);
    auto platforms = new cl_platform_id[platform_number];
    clGetPlatformIDs(platform_number, platforms, &platform_number);
    cl_device_id device_GPU = nullptr, device_GPU_integrated = nullptr, device_CPU = nullptr;
    printf("Devices:\n");
    for (auto i = 0; i < platform_number; ++i) {
        cl_uint device_number;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_number);
        auto devices = new cl_device_id[device_number];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_number, devices, &device_number);
        for (auto j = 0; j < device_number; j++) {
            size_t device_name_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &device_name_size);
            auto device_name = new char[device_name_size];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, device_name_size, device_name, &device_name_size);
            printf("%i) %s\n", i + 1, device_name);
            size_t device_type_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, 0, NULL, &device_type_size);
            auto device_type = new char[device_type_size];
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, device_type_size, (void *) device_type, &device_type_size);
            if ((cl_device_type) (*device_type) == CL_DEVICE_TYPE_CPU) {
                device_CPU = devices[j];
            }
            if ((cl_device_type) (*device_type) == CL_DEVICE_TYPE_GPU) {
                size_t device_vendor_size;
                clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, NULL, &device_vendor_size);
                std::string device_vendor;
                device_vendor.resize(device_vendor_size);

                clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, device_vendor_size, device_vendor.data(), NULL);

                //TODO: Add something for amd?
                if (device_vendor.find("Intel") != std::string::npos) {
                    device_GPU_integrated = devices[j];
                } else {
                    device_GPU = devices[j];
                }
            }
            delete[] device_type;
            delete[] device_name;
        }
        delete[] devices;
    }
    delete[] platforms;
    cl_device_id device = nullptr;
    if (device_GPU != nullptr) device = device_GPU;
    else if (device_GPU_integrated != nullptr) device = device_GPU_integrated;
    else if (device_CPU != nullptr) device = device_CPU;
    else {
        perror("Can't find any devices");
    }
    size_t device_name_size;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &device_name_size);
    auto device_name = new char[device_name_size];
    clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name, &device_name_size);
    printf("Running on:\n%s\n", device_name);

    cl_int error_code;

    cl_context context = clCreateContext(NULL, 1, &device, 0, 0, &error_code);
    if (error_code < 0) {
        perror("ClCreateContext failed");
        return 0;
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error_code);
    if (error_code < 0) {
        perror("clCreateCommandQueue failed");
        return 0;
    }

    FILE *kernel_file = fopen("../mul_matrix.cl", "r");
    if (!kernel_file) {
        perror("Can't open file");
        return 0;
    }
    size_t file_size = 1024 * 20;

    char *program_code = new char[file_size];
    size_t code_len = fread(program_code, 1, file_size, kernel_file);

//    printf("%s", program_code);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &program_code,
                                                   &code_len, &error_code);
    if (error_code < 0) {
        perror("clCreateProgramWithSource failed");
        return 0;
    }
    error_code = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (error_code != 0) {
        perror("kernel file compilation failed");
        return 0;
    } else {
        printf("Kernel file build successfully\n");
    }

    const size_t n = 2048;
    const size_t m = 512;
    const size_t k = 1024;

    const size_t array1_size = n * m * sizeof(float);
    const size_t array2_size = m * k * sizeof(float);
    const size_t array3_size = n * k * sizeof(float);

    auto *array1 = new float[n * m];
    auto *array2 = new float[m * k];
    auto *array_res = new float[n * k];

    fill_matrix(array1, n * m);
    fill_matrix(array2, m * k);

    char* kernel_name = "mul_matrix";

    cl_kernel kernel = clCreateKernel(program, kernel_name, &error_code);
    if (error_code < 0) {
        perror("clCreateKernel failed");
        return 0;
    }
    cl_mem array1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, array1_size, 0, &error_code);
    if (error_code < 0) {
        perror("clCreateBuffer 1 failed");
        return 0;
    }
    cl_mem array2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, array2_size, 0, &error_code);
    if (error_code < 0) {
        perror("clCreateBuffer 2 failed");
        return 0;
    }
    cl_mem array_res_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, array3_size, 0, &error_code);
    if (error_code < 0) {
        perror("clCreateBuffer (result) failed");
        return 0;
    }

    error_code = clEnqueueWriteBuffer(queue, array1_buffer, false, 0, array1_size, array1, 0, 0, 0);
    if (error_code < 0) {
        perror("clEnqueueWriteBuffer 1 failed");
        return 0;
    }
    error_code = clEnqueueWriteBuffer(queue, array2_buffer, true, 0, array2_size, array2, 0, 0, 0);
    if (error_code < 0) {
        perror("clEnqueueWriteBuffer 2 failed");
        return 0;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &array1_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &array2_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &array_res_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &m);
    clSetKernelArg(kernel, 5, sizeof(cl_uint), &k);

    size_t work_offset[] = {0, 0};
    size_t work_size[] = {n, k};
    cl_event run_event;
    clEnqueueNDRangeKernel(queue, kernel, 2, work_offset, work_size, 0, 0, 0, &run_event);
    clEnqueueReadBuffer(queue, array_res_buffer, true, 0, array3_size, array_res, 0, 0, 0);

    cl_ulong t_start = 0, t_end = 0;
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, 0);
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, 0);

    printf("%lu ns elapsed\n", t_end - t_start);


    printf("Please wait, check results...\n");
    if (check_matrix(n, m, k, array1, array2, array_res)) {
        printf("Matrix is OK\n");
    } else {
        printf("Matrix is Bad\n");
    }
    return 0;
}