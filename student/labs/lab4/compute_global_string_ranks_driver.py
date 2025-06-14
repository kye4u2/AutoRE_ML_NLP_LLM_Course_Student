import os

from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from labs.lab4.binary_rank import compute_global_strings_ranks, BinaryRankContext


bcc_file_path = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                             "Blackfyre",
                             "test",
                             "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90f_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc")

def main():
    binary_rank_context = BinaryRankContext.from_bcc_file_path(bcc_file_path)

    string_rank_dict = compute_global_strings_ranks(binary_rank_context)

    # Display the top 10 strings by global rank
    print("\nTop 10 strings by global rank:")
    for index, (string, global_rank) in enumerate(sorted(string_rank_dict.items(), key=lambda x: x[1], reverse=True)[:10]):
        print(f"{index+1})  - String: {string} Global Rank: {global_rank}")


if __name__ == "__main__":
    main()
