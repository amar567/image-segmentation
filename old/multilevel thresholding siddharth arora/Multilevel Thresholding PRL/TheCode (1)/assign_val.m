function new_mat=assign_val(mat,hh,hl,ll,lh,high_assign,low_assign)
[m,n]=size(mat);
for i=1:m
    for j=1:n
        if mat(i,j)>=hl & mat(i,j)<=hh
           mat(i,j)=high_assign;
        end
        if mat(i,j)<=lh & mat(i,j)>=ll
           mat(i,j)=low_assign;
        end
    end
end
new_mat=mat;