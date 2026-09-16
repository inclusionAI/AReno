// Gate activation and optional Q/K normalization are native TPC operations.
#include "recurrent_math.h"
void main(tensor q,tensor k,tensor raw_gate,tensor a_log,tensor dt_bias,
#if ARENO_DIRECTION == 0
          tensor beta,tensor prepared,
#else
          tensor grad_prepared,tensor dq,tensor dk,tensor dg,tensor da,tensor dbias,tensor dbeta,
#endif
          int tokens,int q_heads,int heads,int key_dim,int normalize,int storage_dtype,int gate_dtype,int recurrent,int bounded,
          float lower_bound,float softplus_beta,float softplus_threshold) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for (int row=begin[0]; row<end[0]; ++row) {
        int head=row%heads,qr=(row/heads)*q_heads+head/(heads/q_heads);
        float qnorm=1,knorm=1;
        if (normalize) {
            float qs=1e-6f,ks=1e-6f;
            for (int j=0; j<key_dim; ++j) { float a=rload(q,qr,j),b=rload(k,qr,j); qs+=a*a; ks+=b*b; }
            qnorm=rrsqrt(qs); knorm=rrsqrt(ks);
        }
        float rate=rexp(rload(a_log,0,head));
#if ARENO_DIRECTION == 1
        float qdot=0,kdot=0,rate_grad=0;
        if (normalize) for (int j=0; j<key_dim; ++j) {
            qdot+=rround(rload(grad_prepared,row,j),storage_dtype)*rload(q,qr,j)*qnorm;
            kdot+=rround(rload(grad_prepared,row,key_dim+j),storage_dtype)*rload(k,qr,j)*knorm;
        }
#endif
        for (int j=0; j<key_dim; ++j) {
            float x=rload(raw_gate,row,j)+rload(dt_bias,head,j);
            float sigmoid=rsigmoid(rate*x);
            float gate=bounded ? lower_bound*sigmoid : -rate*rsoftplus(x,softplus_beta,softplus_threshold);
            float decay=rexp(recurrent ? gate : rround(gate,gate_dtype));
#if ARENO_DIRECTION == 0
            float query=rload(q,qr,j)*qnorm,key=rload(k,qr,j)*knorm;
            if (normalize && !recurrent) { query=rround(query,storage_dtype); key=rround(key,storage_dtype); }
            rstore(prepared,row,j,query); rstore(prepared,row,key_dim+j,key); rstore(prepared,row,2*key_dim+j,decay);
#else
            float query_grad=rload(grad_prepared,row,j),key_grad=rload(grad_prepared,row,key_dim+j);
            if (normalize && !recurrent) { query_grad=rround(query_grad,storage_dtype); key_grad=rround(key_grad,storage_dtype); }
            float dquery=qnorm*(query_grad-rload(q,qr,j)*qnorm*qdot);
            float dkey=knorm*(key_grad-rload(k,qr,j)*knorm*kdot);
            radd(dq,qr,j,dquery); radd(dk,qr,j,dkey);
            float dgate=rload(grad_prepared,row,2*key_dim+j)*decay;
            if (!recurrent) dgate=rround(dgate,gate_dtype);
            float dx=bounded ? dgate*lower_bound*sigmoid*(1-sigmoid)*rate
                : -dgate*rate*(x*softplus_beta > softplus_threshold ? 1 : rsigmoid(x*softplus_beta));
            rate_grad+=bounded ? dx*x : dgate*gate;
            rstore(dg,row,j,dx); radd(dbias,head,j,dx);
#endif
        }
#if ARENO_DIRECTION == 0
        float b=rload(beta,row,0); rstore(prepared,row,3*key_dim,recurrent ? rsigmoid(b) : b);
#else
        radd(da,0,head,rate_grad);
        // Training receives already activated beta, just like CUDA chunk_kda.
        rstore(dbeta,row,0,rload(grad_prepared,row,3*key_dim));
#endif
    }
}
