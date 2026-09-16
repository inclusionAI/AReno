// Segmented recurrence. Canonical state matrices are [value, key].
#include "recurrent_math.h"
#include "index_io.h"
void main(tensor q,tensor k,tensor v,tensor initial,tensor decay,tensor offsets,tensor lengths,tensor slots,tensor state_scales,
#if ARENO_KIND == 3
          tensor mask,
#endif
          tensor output,tensor final_state,
#if ARENO_KIND == 2
          tensor snapshots,
#endif
          int tokens,int heads,int hidden,int sequences,int slot_count,int steps,int mask_size,int storage_dtype,float scale) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for (int task=begin[0]; task<end[0]; ++task) {
        int value=task%hidden,head=(task/hidden)%heads,seq=task/(heads*hidden);
        int slot=load_index64(rc(0,seq),slots);
        if (slot < 0 || slot >= slot_count) continue;
#if ARENO_KIND == 1
        int first=seq,length=1;
#elif ARENO_KIND == 2
        int first=seq*steps,length=steps;
#else
        int first=s_i32_ld_g(gen_addr(rc(0,seq),offsets)),length=s_i32_ld_g(gen_addr(rc(0,seq),lengths));
#endif
        float state[512],rate=rload(decay,0,head);
#if ARENO_KIND == 0 || ARENO_KIND == 3
        bool use_state=rload(state_scales,0,seq) > 0;
#else
        bool use_state=true;
#endif
        for (int j=0; j<hidden; ++j) state[j]=use_state ? rload(initial,(slot*heads+head)*hidden+value,j) : 0;
        for (int t=0; t<length; ++t) {
            int row=(first+t)*heads+head;
#if ARENO_KIND == 3
            int depth=-1;
            for (int i=0; i<length; ++i) depth+=s_u8_ld_g(gen_addr(rc(seq*mask_size+t,i),mask));
#else
            float vv=rload(v,row,value),factor=rexp(-rate);
            for (int j=0; j<hidden; ++j) state[j]=state[j]*factor+rload(k,row,j)*vv;
#endif
            float total=0;
            const int split=ARENO_KIND == 1 ? 128 : 32;
            for (int base=0; base<hidden; base+=split) {
                float part=0;
                for (int j=base; j<base+split && j<hidden; ++j) {
                    float state_value=state[j];
#if ARENO_KIND == 3
                    state_value*=rexp(-rate*(depth+1));
                    for (int previous=0; previous<length; ++previous) {
                        if (!s_u8_ld_g(gen_addr(rc(seq*mask_size+t,previous),mask))) continue;
                        int previous_depth=-1;
                        for (int i=0; i<length; ++i) previous_depth+=s_u8_ld_g(gen_addr(rc(seq*mask_size+previous,i),mask));
                        int previous_row=(first+previous)*heads+head;
                        state_value+=rload(k,previous_row,j)*rload(v,previous_row,value)*rexp(-rate*(depth-previous_depth));
                    }
#endif
                    part+=rload(q,row,j)*state_value;
                }
                total+=rround(part*scale,storage_dtype);
            }
            rstore(output,row,value,total);
#if ARENO_KIND == 2
            for (int j=0; j<hidden; ++j) rstore(snapshots,((seq*steps+t)*heads+head)*hidden+value,j,state[j]);
#endif
        }
        for (int j=0; j<hidden; ++j) rstore(final_state,(seq*heads+head)*hidden+value,j,state[j]);
    }
}
