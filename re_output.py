import sys
import json
from os.path import join

from evaluation import output

file_directories = sys.argv[1:]

for fdir in file_directories:
    print(fdir)
    try:
        results = json.load(open(join(fdir, "results.json")))
    except FileNotFoundError:
        print("   skipping (no results.json)")

    output.to_latex_table_row(results, fdir,
                              name=fdir.split("/")[-1].replace("_", " "),
                              italic_ood=False,
                              blank_ood=False,
                              italic_entrop=False,
                              blank_classif=False)

