#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <OpenCL/cl.h>

#define ROWS 200
#define COLS 200

int AMAT[ROWS][COLS];
int BMAT[ROWS][COLS];
int CMAT[ROWS][COLS];

const char *programSource =
"__kernel                                   \n"
"void vecmult(__global int *A,              \n"
"             __global int *B,              \n"
"             int wA,                       \n"
"             int hA,                       \n"
"             int wB,                       \n"
"             int hB,                       \n"
"             __global int *C)              \n"
"{                                          \n"
"  int row=get_global_id(1);                \n"
"  int col=get_global_id(0);                \n"
"  int acum=0;                              \n"
"  for (int i=0;i<wA;i++)                   \n"
"     acum+=A[row*wA+i]*B[i*wB+col];        \n"
"  C[row*wB+col]=acum;                      \n"
"}                                          \n"
;



void MatMultCL(int A[ROWS][COLS], int B[ROWS][COLS], int C[ROWS][COLS]){
    time_t start, end;
    
    size_t datasize = sizeof(int)*ROWS*COLS;
    
    int wA = COLS;
    int hA = ROWS;
    int wB = COLS;
    int hB = ROWS;
    
    cl_int status;
    
    //PASO 1: Identificar e inicializar las plataformas
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    int selectedPlatform = 0;
    int platformIndex;
    
    size_t platformNameLength;
    char *platformName = NULL;
    size_t deviceNameLength;
    char *deviceName = NULL;
    
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    
    //PASO 2: Identificar e inicializar los dispositivos
    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;
    int selectedDevice = 0;
    int deviceIndex = 0;
    
    for (platformIndex = 0; platformIndex < numPlatforms; platformIndex++) {
        platformNameLength = 0;
        platformName = NULL;
        clGetPlatformInfo(platforms[platformIndex], CL_PLATFORM_NAME, 0, 0, &platformNameLength);
        platformName = (char *)malloc(platformNameLength);
        clGetPlatformInfo(platforms[platformIndex], CL_PLATFORM_NAME, platformNameLength, platformName, NULL);
        printf("Datos de la plataforma[%d] %s:\n", platformIndex, platformName);
        free(platformName);
        clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        if (numDevices != 0) {
            deviceNameLength = 0;
            deviceName = NULL;
            devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
            clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, 0, &deviceNameLength);
            deviceName = (char *)malloc(deviceNameLength);
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, deviceNameLength, deviceName, NULL);
            printf("\tDispositivo[0:CPU]=%s (%d)\n", deviceName, numDevices);
            free(deviceName);
            free(devices);
        }
        clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        if (numDevices != 0) {
            deviceNameLength = 0;
            deviceName = NULL;
            devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
            clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, 0, &deviceNameLength);
            deviceName = (char *)malloc(deviceNameLength);
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, deviceNameLength, deviceName, NULL);
            printf("\tDispositivo[0:CPU]=%s (%d)\n", deviceName, numDevices);
            free(deviceName);
            free(devices);
        }
    }
    
    do {
        printf("Que indice de plataforma utilizara?");
        scanf("%d", &selectedPlatform);
    } while ((selectedPlatform < 0) && (selectedPlatform < numPlatforms));
    do {
        printf("Que indice de dispositivo utilizara?");
        scanf("%d", &selectedDevice);
    } while ((selectedDevice < 0) && (selectedDevice < 2));
    
    start = clock();
    switch (selectedDevice) {
        case 0:
            status = clGetDeviceIDs(platforms[selectedPlatform], CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
            devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
            status = clGetDeviceIDs(platforms[selectedPlatform], CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
            break;
        case 1:
            status = clGetDeviceIDs(platforms[selectedPlatform], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
            devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
            status = clGetDeviceIDs(platforms[selectedPlatform], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
            break;
    }
    //Usando el GPU
    //	status=clGetDeviceIDs(platforms[1],CL_DEVICE_TYPE_GPU,0,NULL,&numDevices);
    //Usando el CPU
    //	status=clGetDeviceIDs(platforms[1],CL_DEVICE_TYPE_CPU,0,NULL,&numDevices);
    
    //	devices=(cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    //Usando el GPU
    //	status=clGetDeviceIDs(platforms[1],CL_DEVICE_TYPE_GPU,numDevices,devices,NULL);
    //Usando el CPU
    //	status=clGetDeviceIDs(platforms[1],CL_DEVICE_TYPE_CPU,numDevices,devices,NULL);
    
    //PASO 3: Crear el contexto
    cl_context context = NULL;
    context=clCreateContext(NULL,numDevices,devices,NULL,NULL,&status);
    
    //PASO 4: Crear la fila de ordenes ("command queue")
    cl_command_queue cmdQueue;
    cmdQueue=clCreateCommandQueue(context,devices[0],0,&status);
    
    //PASO 5: Crear los "device buffers"
    cl_mem bufferA;
    cl_mem bufferB;
    cl_mem bufferC;
    bufferA=clCreateBuffer(context,CL_MEM_READ_ONLY,datasize,NULL,&status);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    bufferC=clCreateBuffer(context,CL_MEM_WRITE_ONLY,datasize,NULL,&status);
    
    //PASO 6: Pasar los datos de la memoria del CPU a los "buffers" del dispositivo
    status=clEnqueueWriteBuffer(cmdQueue,bufferA,CL_FALSE,0,datasize,&A[0][0],0,NULL,NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufferB, CL_FALSE, 0, datasize, &B[0][0], 0, NULL, NULL);
    
    //PASO 7: Crear y compilar el programa
    cl_program program=clCreateProgramWithSource(context,1,(const char**)&programSource,NULL,&status);
    status=clBuildProgram(program,numDevices,devices,NULL,NULL,NULL);
    
    //PASO 8: Crear el kernel
    cl_kernel kernel=NULL;
    kernel=clCreateKernel(program,"vecmult",&status);
    
    //PASO 9: Pasar argumentos al kernel
    status=clSetKernelArg(kernel,0,sizeof(cl_mem),&bufferA);
    status|=clSetKernelArg(kernel,1,sizeof(cl_mem),&bufferB);
    status|=clSetKernelArg(kernel,2,sizeof(int),&wA);
    status|=clSetKernelArg(kernel,3,sizeof(int),&hA);
    status|=clSetKernelArg(kernel,4,sizeof(int),&wB);
    status|=clSetKernelArg(kernel,5,sizeof(int),&hB);
    status|=clSetKernelArg(kernel,6,sizeof(cl_mem),&bufferC);
    
    //Paso 10: Configurar la estructura del "work-item"
    size_t globalWorkSize[2]={ROWS,COLS};
    size_t localWorkSize[2]={2,1};
    
    //PASO 11: Enfilar el "kernel" para ejecucion
    status=clEnqueueNDRangeKernel(cmdQueue,kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
    switch (status) {
        case CL_INVALID_PROGRAM_EXECUTABLE:
            printf("ERROR: INVALID_PROGRAM_EXECUTABLE\n");
            break;
        case CL_INVALID_COMMAND_QUEUE:
            printf("ERROR: CL_INVALID_COMMAND_QUEUE\n");
            break;
        case CL_INVALID_KERNEL:
            printf("ERROR: INVALID_KERNEL\n");
            break;
        case CL_INVALID_CONTEXT:
            printf("ERROR: INVALID_CONTEXT\n");
            break;
        case CL_INVALID_KERNEL_ARGS:
            printf("ERROR: INVALID_KERNEL_ARGS\n");
            break;
        case CL_INVALID_WORK_DIMENSION:
            printf("ERROR: INVALID_WORK_DIMENSION\n");
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            printf("ERROR: INVALID_WORK_GROUP_SIZE\n");
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            printf("ERROR: INVALID_WORK_ITEM_SIZE\n");
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            printf("ERROR: CL_INVALID_GLOBAL_OFFSET\n");
            break;
        case CL_OUT_OF_RESOURCES:
            printf("ERROR: OUT_OF_RESOURCES\n");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            printf("ERROR: MEM_OBJECT_ALLOCATION_FAILURE\n");
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            printf("ERROR: INVALID_EVENT_WAIT_LIST\n");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            printf("ERROR: OUT_OF_HOST_MEMORY\n");
            break;
        case CL_SUCCESS:
            printf("SUCCESS!!!\n");
            break;
        default:
            printf("ERROR: !?!?!\n");
            break;
    }
    end=clock();
    //	if (status != CL_SUCCESS)
    //		return(0);
    
    //PASO 12: Transportar el "buffer" de resultado a la memoria del CPU
    clEnqueueReadBuffer(cmdQueue,bufferC,CL_TRUE,0,datasize,&C[0][0],0,NULL,NULL);
    bool result=1;
    for (int r=0;r<ROWS;r++)
        for (int c=0;c<COLS;c++) {
            if(C[r][c]!=ROWS) {
                result=0;
                break;
            }
        }
    if(result) {
        printf("Resultados correctos %ld\n",end-start);
    } else {
        printf("Resultados erroneos\n");
    }
    
    //PASO 13: Liberar recursos OpenCL
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(context);
    
    free(platforms);
    free(devices);
    
}

int main() {
    
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) {
            AMAT[r][c] = 1;
            BMAT[r][c] = 1;
            CMAT[r][c] = -100;
        }
    
    MatMultCL(AMAT,BMAT,CMAT);
    
}
