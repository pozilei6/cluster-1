#goes in file: stringMethods.py
#line: after 232

#--------------------------------------------------------------------------------------------------------
        #additional args for cl-N-str:    nsingl_nzero_col_ind, sizes, Normal=False  ->  get_cl_string()

        #insert after line 232 in stringMethods.py

        origC = nsingl_nzero_col_ind

        # make header clustered cols string
        header_str = " "*4                                                                 
        for cols in cols_ser_decomposition:
            prev_len = len(origC[cols[0]])
            for j in cols:
                j_orig = origC[j]
                header_str += ' '*(2 + (2 - prev_len_char) - 1) + j_orig 
                prev_len = len(j_orig)
            header_str += "  "*(2 - (prev_len - 1))                       
        header_str = header_str[0:-2]
        
        
        # make rows & cols clustered string including row indices and cols indices
        row_cols_cl_str = header_str + "\n\n"
        
        for r_inds in ser_decomposition:
            
            for i in r_inds: # one row string

                row_cols_cl_str +=  ' '*(3 - len(str(sizes[i]) + 'x')) + ' '  + str(sizes[i]) + 'x' + '  '  
                A_i = A_copy[i]
                
                for j_inds in cols_ser_decomposition:
                    
                    for j in j_inds:
                        row_cols_cl_str += '×' + '  ' if A_i[j] == 1 else ' ' * 3         

                        
                    row_cols_cl_str += "    "                                           
                    
                row_cols_cl_str = row_cols_cl_str[0:-2]
                row_cols_cl_str += "\n" #go to next row within same row-cluster
                
            row_cols_cl_str += "\n\n" #go to next row-cluster
            
        row_cols_cl_str = row_cols_cl_str[0:-2]
#--------------------------------------------------------------------------------------------------------
