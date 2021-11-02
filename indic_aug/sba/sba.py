import codecs
import sys
from math import ceil


def sba(input_file_src, input_file_tgt, output_file_src, output_file_tgt, probability):
    with codecs.open(input_file_src, 'r', encoding='utf-16') as f_src_in, \
            codecs.open(input_file_tgt, 'r', encoding='utf-16') as f_tgt_in, \
            codecs.open(output_file_src, 'w', encoding='utf-16') as f_src_out, \
            codecs.open(output_file_tgt, 'w', encoding='utf-16') as f_tgt_out:

        src_file = f_src_in.readlines()
        tar_file = f_tgt_in.readlines()
        for i in range(0, len(src_file), 2):
            if(i+1 < len(src_file)):
                # convert strings to list of strings
                src_word_list_1 = src_file[i].strip().split(" ")
                tar_word_list_1 = tar_file[i].strip().split(" ")
                src_word_list_2 = src_file[i+1].strip().split(" ")
                tar_word_list_2 = tar_file[i+1].strip().split(" ")
                # print(src_word_list_1)
                # calculate the starting and ending words for concatenation of sentences
                src_start = ceil(len(src_word_list_1)*probability)
                src_end = ceil(len(src_word_list_2)*probability)
                tar_start = ceil(len(tar_word_list_1)*probability)
                tar_end = ceil(len(tar_word_list_2)*probability)

                src_combined = ""
                for j in range(src_start, len(src_word_list_1), 1):
                    src_combined += src_word_list_1[j]+" "
                for j in range(0, src_end, 1):
                    src_combined += src_word_list_2[j]+" "

                tar_combined = ""
                for j in range(tar_start, len(tar_word_list_1), 1):
                    tar_combined += tar_word_list_1[j]+" "
                for j in range(0, tar_end, 1):
                    tar_combined += tar_word_list_2[j]+" "
                f_src_out.write(src_combined + "\n")
                f_tgt_out.write(tar_combined + "\n")


if __name__ == "__main__":
    input_file_src = sys.argv[1]
    input_file_tgt = sys.argv[2]
    output_file_src = sys.argv[3]
    output_file_tgt = sys.argv[4]
    noise_proba = float(sys.argv[5])
    sba(input_file_src, input_file_tgt,
        output_file_src, output_file_tgt, noise_proba)
