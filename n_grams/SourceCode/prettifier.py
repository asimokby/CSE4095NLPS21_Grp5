from prettytable import PrettyTable
from os.path import join, exists
from os import getcwd, listdir, sep, mkdir, makedirs


def main():

    src_base = join(getcwd(), "out", "est_stats")
    out_base_prettified = join(getcwd(), "out", "prettified_results")

    for i in range(0,8):

        donem_number = "2" + str(i)
        # "donem_"+donem_number+"_without_stopwords"
        src_path = join(src_base, "donem_"+str(donem_number)+"_without_stopwords")
        all_files = listdir(src_path)

        for f in all_files:
            print(f)
            pt = PrettyTable()
        
            if "_r" in f:
                pt.field_names = ["r", "P_mle"]
            elif "_" in f:
                pt.field_names = ["n-gram", "frequency", "p_est"]
            else:
                pt.field_names = ["n-gram", "frequency", ]
            
            src_file = open(src_path+ sep +f, "r", encoding="UTF-8")
            
            for line in src_file.readlines():
                
                if "_r" in f:
                    row_arr = line.split(",")
                else:
                    l_paranthesis_index = line.rindex(")") + 2
                    sub_string_0 = line[0:l_paranthesis_index]
                    print(sub_string_0)
                    sub_string_1_arr = line[l_paranthesis_index:].split(",") 
                    print(str(sub_string_1_arr))
                    row_arr = []
                    row_arr.append(sub_string_0)
                    row_arr+=sub_string_1_arr

                pt.add_row(row_arr)
            out_base_donem = join(out_base_prettified, "donem_"+str(donem_number)+"_without_stopwords")
            if not exists(out_base_donem):
                makedirs(out_base_donem)
            pt_file = open(out_base_donem+sep+f+"_pt.txt","w", encoding="UTF-8")
            pt_file.write(pt.get_string())
            pt_file.flush()
            pt_file.close()

main()