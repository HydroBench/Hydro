
#ifndef __GPUCODE__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <hmpprt/Grouplet.h>
#include <hmpprt/HostTypes.h>
#include <hmpprt/Context.h>
#include <hmpprt/OpenCLTypes.h>
#include <hmpprt/OpenCLGrid.h>
#include <hmpprt/OpenCLModule.h>
#include <hmpprt/DeviceManager.h>
#include <hmpperr/hmpperr.h>

#include <CL/cl.h>

#ifdef _WIN32
#  define CDLT_API __declspec(dllexport)
#else /* ! _WIN32 */
#  define CDLT_API
#endif /* _WIN32 */

#else

#if defined(CL_KHR_FP64_SUPPORTED) && (defined(CL_VERSION_1_0) || defined(CL_VERSION_1_1))
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#ifdef GLOBAL_ATOMIC_EXTS_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#endif

#ifdef LOCAL_ATOMIC_EXTS_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#endif

#ifdef BYTE_ADDRESSABLE_STORE_EXTS_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
#endif

#endif /* __GPUCODE__ */



#ifndef __GPUCODE__

#else


#endif
#define HMPPCG_SIMD_LENGTH 1

# 5 "<preprocessor>"

#ifndef __GPUCODE__
extern "C" CDLT_API  CDLT_API void __hmpp_acc_region__ibx8cgo0(cl_int narray_1, cl_int Hnxyt_1, cl_double Hgamma, cl_int slices, cl_int Hstep_1, cl_double* qgdnv, cl_double* flux)
;
#endif /* __GPUCODE__ */



# 5 "<preprocessor>"

#ifndef __GPUCODE__
CDLT_API void __hmpp_acc_region__ibx8cgo0_internal_1(cl_int narray_11, cl_int Hnxyt_11, cl_double Hgamma, cl_int slices, cl_int Hstep_11, hmpprt::DevicePtr<hmpprt::MS_OPENCL_GLOB,cl_double>  qgdnv, hmpprt::DevicePtr<hmpprt::MS_OPENCL_GLOB,cl_double>  flux)
;
#endif /* __GPUCODE__ */



# 5 "<preprocessor>"

#ifndef __GPUCODE__
static hmpprt::OpenCLGrid * __hmpp_acc_region__ibx8cgo0_loop2D_1 = 0;
#else
__kernel  void __hmpp_acc_region__ibx8cgo0_loop2D_1(int narray, int Hnxyt, int slices_1, int Hstep, __global double* qgdnv_1, __global double* flux_1, double entho_1);
#endif // __GPUCODE__
# 5 "<preprocessor>"

#ifdef __GPUCODE__

__kernel __attribute__((reqd_work_group_size(32, 4, 1)))  void __hmpp_acc_region__ibx8cgo0_loop2D_1(int narray, int Hnxyt, int slices_1, int Hstep, __global double* qgdnv_1, __global double* flux_1, double entho_1)
{
 # 65 "cmpflx.c"
 double qgdnvID;
 # 63 "cmpflx.c"
 double ekin_1;
 # 63 "cmpflx.c"
 double etot_1;
 # 63 "cmpflx.c"
 int i_1;
 # 58 "cmpflx.c"
 int s_1;
 # 65 "cmpflx.c"
 i_1 = (get_global_id(0));
 # 65 "cmpflx.c"
 if (i_1 > narray - 1)
 {
  # 65 "cmpflx.c"
  goto __hmppcg_label_1;
 }
 # 65 "cmpflx.c"
 s_1 = (get_global_id(1));
 # 65 "cmpflx.c"
 if (s_1 > slices_1 - 1)
 {
  # 65 "cmpflx.c"
  goto __hmppcg_label_1;
 }
 # 65 "cmpflx.c"
 qgdnvID = *(qgdnv_1 + (s_1 * Hnxyt + i_1));
 # 66 "cmpflx.c"
 double qgdnvIU;
 # 66 "cmpflx.c"
 qgdnvIU = *(qgdnv_1 + (Hstep * Hnxyt + s_1 * Hnxyt + i_1));
 # 67 "cmpflx.c"
 double qgdnvIP;
 # 67 "cmpflx.c"
 qgdnvIP = *(qgdnv_1 + (3 * Hstep * Hnxyt + s_1 * Hnxyt + i_1));
 # 68 "cmpflx.c"
 double qgdnvIV;
 # 68 "cmpflx.c"
 qgdnvIV = *(qgdnv_1 + (2 * Hstep * Hnxyt + s_1 * Hnxyt + i_1));
 # 71 "cmpflx.c"
 double massDensity;
 # 71 "cmpflx.c"
 massDensity = qgdnvID * qgdnvIU;
 # 72 "cmpflx.c"
 *(flux_1 + (s_1 * Hnxyt + i_1)) = massDensity;
 # 75 "cmpflx.c"
 *(flux_1 + (Hstep * Hnxyt + s_1 * Hnxyt + i_1)) = massDensity * qgdnvIU + qgdnvIP;
 # 77 "cmpflx.c"
 *(flux_1 + (2 * Hstep * Hnxyt + s_1 * Hnxyt + i_1)) = massDensity * qgdnvIV;
 # 80 "cmpflx.c"
 ekin_1 = (double) 0.5 * qgdnvID * (qgdnvIU * qgdnvIU + qgdnvIV * qgdnvIV);
 # 81 "cmpflx.c"
 etot_1 = qgdnvIP * entho_1 + ekin_1;
 # 83 "cmpflx.c"
 *(flux_1 + (3 * Hstep * Hnxyt + s_1 * Hnxyt + i_1)) = qgdnvIU * (etot_1 + qgdnvIP);
 # 5 "<preprocessor>"
 __hmppcg_label_1:;
}
#endif /* __GPUCODE__ */



# 5 "<preprocessor>"

#ifndef __GPUCODE__
CDLT_API void __hmpp_acc_region__ibx8cgo0_internal_1(cl_int narray_11, cl_int Hnxyt_11, cl_double Hgamma, cl_int slices, cl_int Hstep_11, hmpprt::DevicePtr<hmpprt::MS_OPENCL_GLOB,cl_double>  qgdnv, hmpprt::DevicePtr<hmpprt::MS_OPENCL_GLOB,cl_double>  flux)
{
 # 43 "cmpflx.c"
 cl_double entho;
 # 46 "cmpflx.c"
 entho = (cl_double) 1.0 / (Hgamma - (cl_double) 1.0);
 # 5 "<preprocessor>"
 if (narray_11 - 1 >= 0 & slices - 1 >= 0)
 {
  hmpprt::OpenCLGridCall __hmppcg_call;
  __hmppcg_call.setSizeX((narray_11 - 1) / 32 + 1);
  __hmppcg_call.setSizeY((slices - 1) / 4 + 1);
  __hmppcg_call.setBlockSizeX(32);
  __hmppcg_call.setBlockSizeY(4);
  __hmppcg_call.setWorkDim(2);
  __hmppcg_call.addLocalParameter((hmpprt::s32) (narray_11), "narray");
  __hmppcg_call.addLocalParameter((hmpprt::s32) (Hnxyt_11), "Hnxyt");
  __hmppcg_call.addLocalParameter((hmpprt::s32) (slices), "slices_1");
  __hmppcg_call.addLocalParameter((hmpprt::s32) (Hstep_11), "Hstep");
  __hmppcg_call.addLocalParameter(&qgdnv, 8, "qgdnv_1");
  __hmppcg_call.addLocalParameter(&flux, 8, "flux_1");
  __hmppcg_call.addLocalParameter(&entho, 8, "entho_1");
  __hmppcg_call.launch(__hmpp_acc_region__ibx8cgo0_loop2D_1, hmpprt::Context::getInstance()->getOpenCLDevice());
 }
 ;
}
#endif /* __GPUCODE__ */



# 5 "<preprocessor>"

#ifndef __GPUCODE__
extern "C" CDLT_API  CDLT_API void __hmpp_acc_region__ibx8cgo0(cl_int narray_1, cl_int Hnxyt_1, cl_double Hgamma, cl_int slices, cl_int Hstep_1, cl_double* qgdnv, cl_double* flux)
{
 # 1 "<preprocessor>"
 (__hmpp_acc_region__ibx8cgo0_internal_1(narray_1, Hnxyt_1, Hgamma, slices, Hstep_1, hmpprt::DevicePtr<hmpprt::MS_OPENCL_GLOB,cl_double> (qgdnv), hmpprt::DevicePtr<hmpprt::MS_OPENCL_GLOB,cl_double> (flux)));
}
#endif /* __GPUCODE__ */




#ifndef __GPUCODE__
extern "C" const char * hmpprt_opencl_get_gpu_code();

static hmpprt::OpenCLModule * hmpprt_module = 0;
static int hmpprt_uses = 0;

extern "C" CDLT_API void * hmpprt_init()
{
  try
  {
    if (hmpprt_uses++ == 0)
    {
      hmpprt_module = new hmpprt::OpenCLModule(hmpprt_opencl_get_gpu_code());
      __hmpp_acc_region__ibx8cgo0_loop2D_1 = new hmpprt::OpenCLGrid(hmpprt_module, "__hmpp_acc_region__ibx8cgo0_loop2D_1");

    }
    hmpprt::Context::getInstance()->getGrouplet()->setTarget(hmpprt::OPENCL);
    hmpprt::Context::getInstance()->getGrouplet()->addSignature("__hmpp_acc_region__ibx8cgo0", "prototype __hmpp_acc_region__ibx8cgo0(narray: s32, Hnxyt: s32, Hgamma: double, slices: s32, Hstep: s32, qgdnv: ^openclglob double, flux: ^openclglob double)");

  }
  catch (hmpperr::Error & e)
  {
    return e.clone();
  }
  catch(...)
  {
    fprintf(stderr,"Unexpected error in hmpprt_init()\n");
    abort();
  }
  return 0;
}
#endif /* __GPUCODE__ */

#ifndef __GPUCODE__
extern "C" CDLT_API void * hmpprt_fini()
{
  try
  {
    if (--hmpprt_uses == 0)
    {
      delete __hmpp_acc_region__ibx8cgo0_loop2D_1;

      delete hmpprt_module;
      hmpprt_module = 0;
    }
  }
  catch (hmpperr::Error & e)
  {
    return e.clone();
  }
  catch(...)
  {
    fprintf(stderr,"Unexpected error in hmpprt_fini()\n");
    abort();
  }
  return 0;
}
#endif /* __GPUCODE__ */

// footer
