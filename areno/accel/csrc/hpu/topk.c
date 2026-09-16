#include "tensor_io.h"
#include "index_io.h"

static inline int5 coords(int row,int column) { int5 c={column,row,0,0,0}; return c; }
static inline float maximum(float a,float b) { return a>b ? a : b; }
static inline float exponential(float value) { float64 v=value; v=v_exp_f32(v); return v[0]; }
static inline void insert(float value,int index,float* values,int* indices,int count) {
    for (int i=0; i<count; ++i) if (value>values[i] || (value==values[i] && index<indices[i])) {
        for (int j=count-1; j>i; --j) { values[j]=values[j-1]; indices[j]=indices[j-1]; }
        values[i]=value; indices[i]=index; break;
    }
}

void main(tensor logits,
#if ARENO_KIND == 1
          tensor bias,
#elif ARENO_DIRECTION == 1
          tensor selected, tensor gradient,
#endif
#if ARENO_DIRECTION == 1
          tensor output,
#else
          tensor selected, tensor weights,
#endif
          int tokens,int experts,int top_k,int renormalize,int groups,int top_groups) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    union { unsigned int bits; float value; } negative_inf;
    negative_inf.bits=0xff800000u;
    for (int token=start[0]; token<end[0]; ++token) {
        float probabilities[512];
#if ARENO_KIND == 1
        float route[512], group_values[64];
        int group_indices[64], included[64];
        for (int e=0; e<experts; ++e) {
            float64 x=load_scalar(coords(token,e),logits);
            x=v_reciprocal_f32(1.0f+v_exp_f32(-x));
            probabilities[e]=x[0];
            route[e]=x[0]+load_scalar_f32(coords(0,e),bias);
        }
        for (int g=0; g<groups; ++g) { group_values[g]=negative_inf.value; group_indices[g]=groups; included[g]=0; }
        for (int g=0; g<groups; ++g) {
            float best[16]; int indices[16];
            int count=top_k/top_groups;
            for (int i=0; i<count; ++i) { best[i]=negative_inf.value; indices[i]=experts; }
            for (int e=g*(experts/groups); e<(g+1)*(experts/groups); ++e) insert(route[e],e,best,indices,count);
            float total=0;
            for (int i=0; i<count; ++i) total+=best[i];
            insert(total,g,group_values,group_indices,top_groups);
        }
        for (int i=0; i<top_groups; ++i) included[group_indices[i]]=1;
#else
        float max_value=negative_inf.value,sum=0;
        for (int e=0; e<experts; ++e) max_value=maximum(max_value,load_scalar(coords(token,e),logits));
        for (int e=0; e<experts; ++e) { probabilities[e]=exponential(load_scalar(coords(token,e),logits)-max_value); sum+=probabilities[e]; }
        sum=maximum(sum,1e-20f);
        for (int e=0; e<experts; ++e) probabilities[e]/=sum;
#endif
#if ARENO_DIRECTION == 0
        float best[16]; int indices[16];
        for (int i=0; i<top_k; ++i) { best[i]=negative_inf.value; indices[i]=0; }
        for (int e=0; e<experts; ++e) {
#if ARENO_KIND == 1
            if (included[e/(experts/groups)]) insert(route[e],e,best,indices,top_k);
#else
            insert(probabilities[e],e,best,indices,top_k);
#endif
        }
        float denominator=0;
        for (int i=0; i<top_k; ++i) denominator+=probabilities[indices[i]];
        denominator=maximum(denominator,1e-20f);
        for (int i=0; i<top_k; ++i) {
            store_index64(coords(token,i),selected,indices[i]);
            float value=probabilities[indices[i]];
            if (renormalize || ARENO_KIND == 1) value/=denominator;
            s_f32_st_g(gen_addr(coords(token,i),weights),value);
        }
#else
        float derivatives[512], selected_sum=0, weighted_grad=0, dot=0;
        for (int e=0; e<experts; ++e) derivatives[e]=0;
        for (int i=0; i<top_k; ++i) {
            int index=load_index64(coords(token,i),selected);
            selected_sum+=probabilities[index];
            weighted_grad+=load_scalar_f32(coords(token,i),gradient)*probabilities[index];
        }
        selected_sum=maximum(selected_sum,1e-20f);
        for (int i=0; i<top_k; ++i) {
            int index=load_index64(coords(token,i),selected);
            float g=load_scalar_f32(coords(token,i),gradient);
            float dp=renormalize ? (g*selected_sum-weighted_grad)/(selected_sum*selected_sum) : g;
            derivatives[index]+=dp; dot+=dp*probabilities[index];
        }
        for (int e=0; e<experts; ++e) store_scalar(coords(token,e),output,probabilities[e]*(derivatives[e]-dot));
#endif
    }
}
