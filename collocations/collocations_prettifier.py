from prettytable import PrettyTable
from os.path import join, exists
from os import getcwd, listdir, sep, mkdir, makedirs


def main():

    src_base = join(getcwd(), "collocations","results")
    out_base_prettified = join(getcwd(), "collocations", "prettified_results")

    for i in range(0,8):

        donem_number = "2" + str(i)
        # "donem_"+donem_number+"_without_stopwords"
        src_path = join(src_base, "donem_"+str(donem_number))
        all_files = listdir(src_path)

        for f in all_files:
            print(f)
            pt = PrettyTable()
    
            src_file = open(src_path+ sep +f, "r", encoding="UTF-8")

            headers = src_file.readline().strip().split(",")

            pt.field_names = headers

            count = 0
            for line in src_file.readlines():
                row_arr = []
                line_split = line.split(",")
                if "freq" in f:
                    row_arr = [line_split[0], line_split[1]+","+line_split[2],line_split[3]]
                else:
                    row_arr = [line_split[0], line_split[1],line_split[2],line_split[3],line_split[4]+","+line_split[5]]
                
                pt.add_row(row_arr)

                count+=1
                if count == 10:
                    break

            out_base_donem = join(out_base_prettified, "donem_"+str(donem_number))
            
            if not exists(out_base_donem):
                makedirs(out_base_donem)
            
            pt_file = open(out_base_donem+sep+f+"_pt.txt","w", encoding="UTF-8")
            pt_file.write(pt.get_string())
            pt_file.flush()
            pt_file.close()

main()