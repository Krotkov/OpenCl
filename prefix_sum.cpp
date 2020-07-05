//
// Created by kranya on 05.07.2020.
//


#include <CL/opencl.h>
#include <cstdio>
#include <iostream>

void fill_array(float *ptr, size_t cnt) {
    srand(time(NULL));
    for (size_t i = 0; i < cnt; ++i) {
        ptr[i] = (float) rand() / RAND_MAX * 500.0 / cnt;
    }
}

bool check_array(size_t n, float *a, float *b) {
    auto *res = new float[n]();
    res[0] = a[0];
    for (size_t i = 1; i < n; ++i) {
        res[i] = a[i] + res[i - 1];
        float delta = res[i] - b[i];
        if (abs(delta) >= 0.1) {
            delete[] res;
            return false;
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

                if (device_vendor.find("NVIDIA") != std::string::npos ||
                    device_vendor.find("AMD") != std::string::npos ||
                    device_vendor.find("nvidia") != std::string::npos ||
                    device_vendor.find("amd") != std::string::npos) {
                    device_GPU = devices[j];
                } else {
                    device_GPU_integrated = devices[j];
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

    cl_ulong max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &max_work_group_size,
                    nullptr);
    printf("Max work group size: %lu\n", max_work_group_size);

    // Base version
    size_t array_length = max_work_group_size;

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

    FILE *kernel_file = fopen("prefix_sum.cl", "r");
    if (!kernel_file) {
        perror("Can't open kernel file");
        return 0;
    }
    size_t file_size = 1024 * 3;
    char *program_code = new char[file_size];

    size_t code_len;
    code_len = fread(program_code, 1, file_size, kernel_file);

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

    const size_t n = array_length;

    const size_t array_size = n * sizeof(float);

    auto *array1 = new float[n];
    auto *array_res = new float[n];
    auto *array_tmp = new float[2*n];

    fill_array(array1, n);
    fill_array(array_res, n);

    char *kernel_name = "prefix_sum";

    cl_kernel kernel = clCreateKernel(program, kernel_name, &error_code);
    if (error_code < 0) {
        perror("clCreateKernel failed");
        return 0;
    }
    cl_mem array1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, array_size, 0, &error_code);
    if (error_code < 0) {
        perror("clCreateBuffer 1 failed");
        return 0;
    }
    cl_mem array_res_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, array_size, 0, &error_code);
    if (error_code < 0) {
        perror("clCreateBuffer res failed");
        return 0;
    }

    cl_mem array_tmp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*array_size, 0, &error_code);
    if (error_code < 0) {
        perror("clCreateBuffer tmp failed");
        return 0;
    }

    error_code = clEnqueueWriteBuffer(queue, array1_buffer, false, 0, array_size, array1, 0, 0, 0);
    if (error_code < 0) {
        perror("clEnqueueWriteBuffer 1 failed");
        return 0;
    }
    error_code = clEnqueueWriteBuffer(queue, array_res_buffer, false, 0, array_size, array_res, 0, 0, 0);
    if (error_code < 0) {
        perror("clEnqueueWriteBuffer res failed");
        return 0;
    }

    error_code = clEnqueueWriteBuffer(queue, array_tmp_buffer, false, 0, 2*array_size, array_tmp, 0, 0, 0);
    if (error_code < 0) {
        perror("clEnqueueWriteBuffer tmp failed");
        return 0;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &array1_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &array_res_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &array_tmp_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);

    size_t work_size[] = {array_length};
    size_t local_group_size[] = {array_length};
    cl_event run_event;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, local_group_size, 0, 0, &run_event);
    clEnqueueReadBuffer(queue, array_res_buffer, true, 0, array_size, array_res, 0, 0, 0);

    cl_ulong t_start = 0, t_end = 0;
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, 0);
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, 0);

    printf("%lu ns elapsed\n", t_end - t_start);

    printf("Please wait, check the result...\n");
    if (check_array(n, array1, array_res)) {
        printf("Prefix sum is OK\n");
    } else {
        printf("Prefix sum is Bad\n");
    }

    delete[] array1;
    delete[] array_res;
    delete[] program_code;
    delete[] device_name;
    return 0;
}