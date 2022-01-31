# Performance mesurements of the various HydroC implementations

The input file used for these runs is input6000.nml in this folder.

**FOM** is MC/s (millions of cells updated by second).

256M/64c/2t = 256 MPI ranks on 32 cores using 2 threads per rank. 

The focus here is on OpenMP / CUDA / OpenCL, even if the code is MPI enabled.

|test case|6000x6000| | | | | | | |laptop|laptop| | | |
|---|---------|-|-|-|-|-|-|-|------|-|-|-|-|
| | |sandybridge|haswell|skylake|knl DDR|knl MCDRAM|thx2|rome|i5-8250U|gen9|v100|a100|mi100
|||||||||||||
|**omphydro**|Bande 1024|**17.47**|**25.53**|**31.81**|**23.38**||||**3.64**|**7.07**|**255.20**|**418.87**|**216**
|offload||16c/16t|32c/64t|24c/24t|64c/128t||||icx, 4c/8t|||nvhpc 21.5|
||||||||||||||
|**cpphydro**|Tile 60|**56.40**|**61.33**|**101.65**|**30.59**|**108.89**|**79.79**|**110.47**|**5.47**|N/A|N/A|N/A|N/A
|||16c/16t|32c/64t|24c/24t|64c/128t|64c/256t|32c/128t|128c/128t|icx 2021.3||||
||||||**129.91**|**172.96**|||**10.63**||||
||||||128M/64c/2t|128M/64c/2t|||icc||||
||||||**216.27**|390.789**|||**10.57**||||
||||||256M/64c/1t|256M/64c/4t|||icx 2021.4||||
|**cuhydro**|Bande 1024|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|**338.80**|**670.00**|N/A
||||||||||||Max 350|Max 700|
||||||||||||||
|**hiphydro**|Bande 1024|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A||**668.50**|**383.30**|
|||||||||||||via hipcc|Max 407
||||||||||||||
|**oclhydro**|Bande 1024|OpenCL not installed|OpenCL not installed|OpenCL not installed|OpenCL not installed|OpenCL not installed|OpenCL not installed|OpenCL not installed|**6.0**|**7.9**|**312.2**|**586.9**|**348.8**|
||||||||||||Max 358|Max 730|
|**HydroSyclMPI**|tilesize=1020||||||||**4.71**|**8.18**||||
|||||||||||dpc++ 2021.4||||
|**cu->dpc++**|tilesize=1020|||||||||**7.9**||||
|||||||||||dpc++ 2021.4||||
