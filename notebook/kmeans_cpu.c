/* kmeans_cpu.c  -- sequential reference implementation (same init rule:
 * first K points; same convergence rule) used to validate the GPU output
 * and to measure CPU baseline time.  Build: gcc -O3 -o kmeans_cpu kmeans_cpu.c -lm
 * Usage: ./kmeans_cpu --data points.bin --k 8 [--iters 100] [--tol 1e-4]
 *                     [--out-assign a.bin] [--out-cent c.bin]            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static double now_ms(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1e3*(double)c.QuadPart/(double)f.QuadPart; }
  #define _CRT_SECURE_NO_WARNINGS
#else
  static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
#endif

int main(int argc, char** argv){
    const char *data=NULL,*oa="assign_cpu.bin",*oc="centroids_cpu.bin"; int K=0,max_iters=100; float tol=1e-4f;
    for(int a=1;a<argc;a++){
        if(!strcmp(argv[a],"--data")) data=argv[++a];
        else if(!strcmp(argv[a],"--k")) K=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--iters")) max_iters=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--tol")) tol=(float)atof(argv[++a]);
        else if(!strcmp(argv[a],"--out-assign")) oa=argv[++a];
        else if(!strcmp(argv[a],"--out-cent")) oc=argv[++a];
    }
    if(!data||K<=0){fprintf(stderr,"usage: --data f.bin --k K\n");return 1;}
    FILE* f=fopen(data,"rb"); if(!f){perror("data");return 1;}
    int N,D; if(fread(&N,4,1,f)!=1||fread(&D,4,1,f)!=1) return 1;
    float* P=malloc((size_t)N*D*4); if(fread(P,4,(size_t)N*D,f)!=(size_t)N*D) return 1; fclose(f);
    float* C=malloc((size_t)K*D*4); memcpy(C,P,(size_t)K*D*4);
    float* S=malloc((size_t)K*D*4); int* cnt=malloc(K*4); int* A=malloc((size_t)N*4);
    for(int i=0;i<N;i++) A[i]=-1;
    double t0=now_ms(); int it, changed=0; float maxshift=0;
    for(it=1;it<=max_iters;it++){
        changed=0; memset(S,0,(size_t)K*D*4); memset(cnt,0,K*4);
        for(int i=0;i<N;i++){
            const float* p=P+(size_t)i*D; float best=FLT_MAX; int bk=0;
            for(int k=0;k<K;k++){ float d=0; for(int j=0;j<D;j++){float df=p[j]-C[k*D+j]; d=fmaf(df,df,d);} if(d<best){best=d;bk=k;} }
            if(A[i]!=bk){A[i]=bk;changed++;}
            for(int j=0;j<D;j++) S[bk*D+j]+=p[j]; cnt[bk]++;
        }
        maxshift=0;
        for(int k=0;k<K;k++) if(cnt[k]>0){ float sh=0; for(int j=0;j<D;j++){ float nv=S[k*D+j]/cnt[k]; float df=nv-C[k*D+j]; sh+=df*df; C[k*D+j]=nv;} if(sh>maxshift)maxshift=sh; }
        if(changed==0||maxshift<tol*tol) break;
    }
    if(it>max_iters) it=max_iters;
    double ms=now_ms()-t0; double inertia=0;
    for(int i=0;i<N;i++) for(int j=0;j<D;j++){double df=P[(size_t)i*D+j]-C[A[i]*D+j]; inertia+=df*df;}
    f=fopen(oa,"wb"); fwrite(&N,4,1,f); fwrite(A,4,N,f); fclose(f);
    f=fopen(oc,"wb"); fwrite(&K,4,1,f); fwrite(&D,4,1,f); fwrite(C,4,(size_t)K*D,f); fclose(f);
    printf("RESULT mode=cpu N=%d D=%d K=%d iters=%d changed_last=%d max_shift=%.3e time_ms=%.3f inertia=%.6e\n",N,D,K,it,changed,sqrtf(maxshift),ms,inertia);
    return 0;
}
