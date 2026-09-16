// One program owns a sequence/head/value column; all recurrence math is FP32.
#include "recurrent_math.h"
#include "index_io.h"
void main(tensor prepared,tensor v,
#if ARENO_DIRECTION == 0
          tensor initial,
#else
          tensor history,tensor grad_output,tensor grad_final,
#endif
          tensor cu_seqlens,tensor indices,
#if ARENO_DIRECTION == 0
          tensor output,tensor final_state,tensor history,
#else
          tensor grad_prepared,tensor grad_v,tensor grad_initial,
#endif
          int tokens,int heads,int key_dim,int value_dim,int sequences,int slots,int save_history,float scale) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for (int task=begin[0]; task<end[0]; ++task) {
        int value=task%value_dim,head=(task/value_dim)%heads,sequence=task/(value_dim*heads);
        int first=s_i32_ld_g(gen_addr(rc(0,sequence),cu_seqlens)),last=s_i32_ld_g(gen_addr(rc(0,sequence+1),cu_seqlens));
        int slot=load_index64(rc(0,sequence),indices);
        float state[512];
        int sr=(sequence*heads+head)*value_dim+value;
#if ARENO_DIRECTION == 0
        for (int j=0; j<key_dim; ++j) state[j]=slot >= 0 && slot < slots ? rload(initial,(slot*heads+head)*value_dim+value,j) : 0;
        for (int token=first; token<last; ++token) {
            int row=token*heads+head; float projection=0;
            for (int j=0; j<key_dim; ++j) {
                if (save_history) rstore(history,row*value_dim+value,j,state[j]);
                state[j]*=rload(prepared,row,2*key_dim+j);
                projection+=state[j]*rload(prepared,row,key_dim+j);
            }
            float update=(rload(v,row,value)-projection)*rload(prepared,row,3*key_dim),out=0;
            for (int j=0; j<key_dim; ++j) {
                state[j]+=rload(prepared,row,key_dim+j)*update;
                out+=state[j]*rload(prepared,row,j);
            }
            rstore(output,row,value,out*scale);
        }
        for (int j=0; j<key_dim; ++j) rstore(final_state,sr,j,state[j]);
#else
        for (int j=0; j<key_dim; ++j) state[j]=rload(grad_final,sr,j);
        for (int token=last-1; token>=first; --token) {
            int row=token*heads+head; float projection=0;
            float go=rload(grad_output,row,value)*scale,beta=rload(prepared,row,3*key_dim);
            for (int j=0; j<key_dim; ++j)
                projection+=rload(history,row*value_dim+value,j)*rload(prepared,row,2*key_dim+j)*rload(prepared,row,key_dim+j);
            float residual=rload(v,row,value)-projection,update=residual*beta,du=0;
            for (int j=0; j<key_dim; ++j) {
                float key=rload(prepared,row,key_dim+j);
                float next=rload(history,row*value_dim+value,j)*rload(prepared,row,2*key_dim+j)+key*update;
                radd(grad_prepared,row,j,go*next);
                state[j]+=go*rload(prepared,row,j); du+=state[j]*key;
            }
            rstore(grad_v,row,value,du*beta); radd(grad_prepared,row,3*key_dim,du*residual);
            for (int j=0; j<key_dim; ++j) {
                float old=rload(history,row*value_dim+value,j),decay=rload(prepared,row,2*key_dim+j);
                float dp=state[j]-rload(prepared,row,key_dim+j)*du*beta;
                radd(grad_prepared,row,key_dim+j,state[j]*update-old*decay*du*beta);
                radd(grad_prepared,row,2*key_dim+j,dp*old);
                state[j]=dp*decay;
            }
        }
        if (slot >= 0 && slot < slots) for (int j=0; j<key_dim; ++j) radd(grad_initial,(slot*heads+head)*value_dim+value,j,state[j]);
#endif
    }
}
