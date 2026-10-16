// SPDX-License-Identifier: MIT
#pragma once
#include "opus_gemm_traits_a8w8_ptpc_bpreshuffle_gfx950.cuh"
#ifdef __HIP_DEVICE_COMPILE__
__device__ constexpr int c18_swizzle(int row,int col){int off=row*128+col;return off^(((off%2048)>>8)<<4);}
__device__ constexpr int c18_a_global(int wave,int lane,int step,int K){
 int row=lane/8+wave*8+step*64,col=(lane%8)*16;
 int off=c18_swizzle(row,col);return (off/128)*K+off%128;
}
__device__ constexpr int c18_b_global(int wave,int lane,int step,int K){
 int row=lane%8+wave*8+step*64,col=(lane/8)*16;
 return(row/16)*K*16+(row%16)*16+(col/64)*1024+((col%64)/16)*256+col%16;
}
__device__ constexpr int c18_b_lds(int row,int col){return(row/8)*1024+(row%8)*16+(col/16)*128+col%16;}
__device__ constexpr int c18_dst(int wave,int lane,int step){return wave*1024+step*8192+lane*16;}
__device__ constexpr int c18_pid_remap(int id,int mt,int nt){
 int total=mt*nt,xcd=id%8,intra=id/8,base=total/8,extra=total%8;
 int remap=xcd*base+(xcd<extra?xcd:extra)+intra;
 int ng=4*nt,group=remap/ng,ig=remap%ng,firstm=group*4,gm=mt-firstm;
 if(gm>4)gm=4;
 int pn=ig/gm,pm=firstm+ig%gm;return pm*nt+pn;
}
template<typename Atom,int W,int L> __device__ constexpr bool c18_verify(){
 using namespace opus;Atom mma;
 auto pa=opus::make_tuple(L%16,L/16),pc=opus::make_tuple(L/16,L%16);
 auto am=layout_to_offsets<1>(mma.template layout_a<1>(opus::make_tuple(1_I,0_I),pa));
 auto ak=layout_to_offsets<1>(mma.template layout_a<1>(opus::make_tuple(0_I,1_I),pa));
 auto bn=layout_to_offsets<1>(mma.template layout_b<1>(opus::make_tuple(1_I,0_I),pa));
 auto bk=layout_to_offsets<1>(mma.template layout_b<1>(opus::make_tuple(0_I,1_I),pa));
 auto cm=layout_to_offsets<1>(mma.template layout_c<1>(opus::make_tuple(1_I,0_I),pc));
 auto cn=layout_to_offsets<1>(mma.template layout_c<1>(opus::make_tuple(0_I,1_I),pc));
 static_assert(Atom::elem_a==32&&Atom::elem_b==32&&Atom::elem_c==4);
 for(int e=0;e<32;++e)if(am[e]!=L%16||ak[e]!=(e/16)*64+(L/16)*16+e%16||bn[e]!=L%16||bk[e]!=ak[e])return false;
 for(int e=0;e<4;++e)if(cm[e]!=(L/16)*4+e||cn[e]!=L%16)return false;

 for(int em=0;em<4;++em)for(int e=0;e<32;++e){
  int m=(W/4)*64+em*16+am[e],k=ak[e],off=c18_swizzle(m,k);
  int wl=(off%1024)/16,ww=(off/1024)%8,st=off/8192,b=off%16;
  int go=c18_a_global(ww,wl,st,8192)+b;
  if(c18_dst(ww,wl,st)+b!=off||go/8192!=m||go%8192!=k)return false;
  if(off!=c18_swizzle((W/4)*64+L%16,(L/16)*16+(e/16)*64)+em*2048+e%16)return false;
 }
 for(int en=0;en<2;++en)for(int e=0;e<32;++e){
  int n=(W%4)*32+en*16+bn[e],k=bk[e],off=c18_b_lds(n,k);
  int wl=(off%1024)/16,ww=(off/1024)%8,st=off/8192,b=off%16;
  int go=c18_b_global(ww,wl,st,8192)+b;
  int canonical=(n/16)*8192*16+(k/32)*512+((k%32)/16)*256+(n%16)*16+k%16;
  if(c18_dst(ww,wl,st)+b!=off||go!=canonical)return false;
  if(off!=c18_b_lds((W%4)*32+L%16,(L/16)*16)+en*2048+(e/16)*512+e%16)return false;
 }
 for(int st=0;st<2;++st){
  int off=c18_dst(W,L,st),ga=c18_a_global(W,L,st,8192);
  if(ga/8192!=L/8+W*8+st*64||off<0||off+16>16384)return false;
 }
 return true;
}
template<typename V> __device__ __forceinline__ void c18_pin(V& v){asm volatile("" : "+v"(v) :: "memory");}
template<int COUNT> __device__ __forceinline__ void c18_wait_barrier(){
 asm volatile("s_waitcnt vmcnt(%0)\ns_barrier"::"n"(COUNT):"memory");
 __builtin_amdgcn_sched_barrier(0);
}
#endif

template<int K,bool SCALE_ORDER_COMPAT> __global__ __launch_bounds__(512,2)
void gemm_ptpc_c18_fly_gfx950(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 a){
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
 using namespace opus;
 constexpr int KI=K/128;static_assert(KI>=2);
 int lane=thread_id_x()%64,wave=__builtin_amdgcn_readfirstlane(thread_id_x()/64),wm=wave/4,wn=wave%4;
 int nt=(a.n+255)/256,mt=(a.m+255)/256;
 int pid=c18_pid_remap(block_id_x(),mt,nt),row=(pid/nt)*256,col=(pid%nt)*256;
 auto gA=make_gmem(reinterpret_cast<const fp8_t*>(a.ptr_a),a.m*K);
 auto gB=make_gmem(reinterpret_cast<const fp8_t*>(a.ptr_b),a.n*K);
 auto gC=make_gmem(reinterpret_cast<bf16_t*>(a.ptr_c),a.m*a.n*2);
 auto gSA=make_gmem(reinterpret_cast<const float*>(a.ptr_sa),a.m*4);
 auto gSB=make_gmem(reinterpret_cast<const float*>(a.ptr_sb),a.n*4);
 __shared__ __align__(16) fp8_t ldsA[4][16384],ldsB[4][16384];
 auto mma=make_mfma<fp8_t,fp8_t,float>(seq<16,16,128>{});
 using Atom=decltype(mma);
 static_ford<8,64>([](auto wi,auto li){static_assert(c18_verify<Atom,decltype(wi)::value,decltype(li)::value>(),"OPUS MFMA atom and LDS writer-reader mapping must verify");});
 typename Atom::vtype_c acc[4][8];
 static_ford<4,8>([&](auto q,auto x){clear(acc[decltype(q)::value][decltype(x)::value]);});
 auto issueA=[&](fp8_t*ptr,int hm,int tk){
  static_ford<2>([&](auto st){constexpr int step=decltype(st)::value;
   auto dst=reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(ptr)+wave*1024+step*8192);
   gA.template async_load<16>(dst,(row+hm*128)*K+c18_a_global(wave,lane,step,K),tk*128);
  });
 };
 auto issueB=[&](fp8_t*ptr,int hn,int tk){
  static_ford<2>([&](auto st){constexpr int step=decltype(st)::value;
   auto dst=reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(ptr)+wave*1024+step*8192);
   gB.template async_load<16>(dst,(col+hn*128)*K+c18_b_global(wave,lane,step,K),tk*2048);
  });
 };
 struct AF{typename Atom::vtype_a v[4];};struct BF{typename Atom::vtype_b v[2];};

 unsigned int abase=static_cast<unsigned int>(reinterpret_cast<__UINTPTR_TYPE__>(ldsA))+c18_swizzle(wm*64+lane%16,(lane/16)*16);
 unsigned int bbase=static_cast<unsigned int>(reinterpret_cast<__UINTPTR_TYPE__>(ldsB))+c18_b_lds(wn*32+lane%16,(lane/16)*16);
 unsigned int abase_hi=static_cast<unsigned int>(reinterpret_cast<__UINTPTR_TYPE__>(ldsA))+c18_swizzle(wm*64+lane%16,(lane/16)*16+64);
 c18_pin(abase);c18_pin(abase_hi);c18_pin(bbase);
 auto readA=[&](auto slot){constexpr int S=decltype(slot)::value;AF f;
  auto sm=make_smem(reinterpret_cast<fp8_t*>(static_cast<__UINTPTR_TYPE__>(abase)));
  auto sh=make_smem(reinterpret_cast<fp8_t*>(static_cast<__UINTPTR_TYPE__>(abase_hi)));
  static_ford<4>([&](auto ii){constexpr int i=decltype(ii)::value;
   auto lo=sm.template load<16>(S*16384+i*2048);
   auto hi=sh.template load<16>(S*16384+i*2048);
   set_slice(f.v[i],lo,0_I,16_I);set_slice(f.v[i],hi,16_I,32_I);
  });return f;
 };
 auto readB=[&](auto slot){constexpr int S=decltype(slot)::value;BF f;
  auto sm=make_smem(reinterpret_cast<fp8_t*>(static_cast<__UINTPTR_TYPE__>(bbase)));
  static_ford<2>([&](auto jj){constexpr int j=decltype(jj)::value;
   auto lo=sm.template load<16>(S*16384+j*2048);
   auto hi=sm.template load<16>(S*16384+j*2048+512);
   set_slice(f.v[j],lo,0_I,16_I);set_slice(f.v[j],hi,16_I,32_I);
  });return f;
 };
 auto compute=[&](auto q,const AF&af,const BF&bf){constexpr int Q=decltype(q)::value;
  s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_sched_barrier(0);__builtin_amdgcn_s_setprio(1);
  static_ford<4,2>([&](auto ii,auto jj){constexpr int i=decltype(ii)::value,j=decltype(jj)::value;
   acc[Q][i*2+j]=mma(af.v[i],bf.v[j],acc[Q][i*2+j],127,127);
  });
  static_ford<8>([&](auto xx){constexpr int x=decltype(xx)::value;c18_pin(acc[Q][x]);});
  __builtin_amdgcn_s_setprio(0);__builtin_amdgcn_sched_barrier(0);__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);
 };
 issueB(ldsB[0],0,0);issueA(ldsA[0],0,0);issueB(ldsB[1],1,0);issueA(ldsA[1],1,0);
 if(wm==1){__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);}
 c18_wait_barrier<4>();
 issueB(ldsB[2],0,1);issueA(ldsA[2],0,1);issueB(ldsB[3],1,1);
 c18_wait_barrier<6>();
 static_ford<KI-2>([&](auto kk){constexpr int k=decltype(kk)::value,cur=(k%2)*2,next=((k+1)%2)*2;
  auto b0=readB(number<cur>{});auto a0=readA(number<cur>{});issueA(ldsA[next+1],1,k+1);
  s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);
  compute(0_I,a0,b0);
  auto b1=readB(number<cur+1>{});issueB(ldsB[cur],0,k+2);__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);compute(1_I,a0,b1);
  auto a1=readA(number<cur+1>{});issueA(ldsA[cur],0,k+2);__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);compute(2_I,a1,b0);
  issueB(ldsB[cur+1],1,k+2);c18_wait_barrier<6>();compute(3_I,a1,b1);
 });
 constexpr int cur=((KI-2)%2)*2,next=((KI-1)%2)*2;
 auto b0=readB(number<cur>{});auto a0=readA(number<cur>{});__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);compute(0_I,a0,b0);
 auto b1=readB(number<cur+1>{});__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);compute(1_I,a0,b1);
 auto a1=readA(number<cur+1>{});issueA(ldsA[next+1],1,KI-1);
 c18_wait_barrier<4>();
 compute(2_I,a1,b0);
 auto lastb0=readB(number<next>{});__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);compute(3_I,a1,b1);
 auto lasta0=readA(number<next>{});c18_wait_barrier<0>();compute(0_I,lasta0,lastb0);
 auto lastb1=readB(number<next+1>{});__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);compute(1_I,lasta0,lastb1);
 auto lasta1=readA(number<next+1>{});__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);
 s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_sched_barrier(0);__builtin_amdgcn_s_setprio(1);
 static_ford<4,2>([&](auto ii,auto jj){constexpr int i=decltype(ii)::value,j=decltype(jj)::value;
  acc[2][i*2+j]=mma(lasta1.v[i],lastb0.v[j],acc[2][i*2+j],127,127);
  acc[3][i*2+j]=mma(lasta1.v[i],lastb1.v[j],acc[3][i*2+j],127,127);
 });
 static_ford<8>([&](auto xx){constexpr int x=decltype(xx)::value;c18_pin(acc[2][x]);c18_pin(acc[3][x]);});
 __builtin_amdgcn_s_setprio(0);__builtin_amdgcn_sched_barrier(0);__builtin_amdgcn_s_barrier();__builtin_amdgcn_sched_barrier(0);

 int et=thread_id_x(),er=row,ecol=col;
 c18_pin(et);asm volatile("" : "+s"(er), "+s"(ecol) :: "memory");
 int el=et%64,ewm=(et/64)/4,ewn=(et/64)%4;
 static_ford<4>([&](auto qq){constexpr int q=decltype(qq)::value,hm=q/2,hn=q%2;
  vector_t<float,4> sav[4];float sbv[2];
  static_ford<4>([&](auto ii){constexpr int i=decltype(ii)::value;sav[i]=gSA.template load<4>(er+hm*128+ewm*64+i*16+(el/16)*4);});
  static_ford<2>([&](auto jj){constexpr int j=decltype(jj)::value;sbv[j]=gSB.template load<1>(ecol+hn*128+ewn*32+j*16+el%16)[0];});
  s_waitcnt_vmcnt(0_I);
  static_ford<4,2,4>([&](auto ii,auto jj,auto cc){constexpr int i=decltype(ii)::value,j=decltype(jj)::value,c=decltype(cc)::value;
   float scaled;if constexpr(SCALE_ORDER_COMPAT)scaled=acc[q][i*2+j][c]*(sav[i][c]*sbv[j]);else scaled=(acc[q][i*2+j][c]*sav[i][c])*sbv[j];
   vector_t<bf16_t,1> v;v[0]=static_cast<bf16_t>(scaled);
   int m=er+hm*128+ewm*64+i*16+(el/16)*4+c,n=ecol+hn*128+ewn*32+j*16+el%16;
   int off=(m<a.m&&n<a.n)?m*a.n+n:a.m*a.n;
   gC.template store<1>(v,off);
  });
 });
#endif
#endif
}
