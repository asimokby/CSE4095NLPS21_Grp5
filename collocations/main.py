from CollocationsByFrequency import CollocationsByFrequency


def main():

    # Method 1: Frequency 
    collocation_by_frquency = CollocationsByFrequency()

    # main loop
    donem_nums = range(20, 21)
    for donem_num in donem_nums:
        collocations = collocation_by_frquency.get_collocations(donem_num)
        print(collocations)

main()
