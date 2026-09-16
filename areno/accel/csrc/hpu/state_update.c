#include "tensor_io.h"
#include "index_io.h"
void main(tensor old_state,tensor new_state,tensor indices,tensor output,int slots,int sequences,int width) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for (int slot=begin[1]; slot<end[1]; ++slot) for (int block=begin[0]; block<end[0]; ++block) {
        int selected=-1;
        for (int seq=0; seq<sequences; ++seq) {
            int5 idx={seq,0,0,0,0};
            if (load_index64(idx,indices) == slot) selected=seq;
        }
        int5 source={block*128,selected >= 0 ? selected : slot,0,0,0},target={block*128,slot,0,0,0};
        if (selected >= 0) copy_storage(source,new_state,target,output);
        else copy_storage(source,old_state,target,output);
    }
}
