import json
from os.path import join
import numpy as np

def to_latex_table_row(results_dict, out_dir, name=""):
    name = name.replace("_", " ")
    outfile = open(join(out_dir, 'results.tex'), 'w')

    outfile.write("{:>14s} &{:>14.2f} &{:>14.2f} &\n".format(name,
                                                     results_dict['test_metrics']['acc'],
                                                     results_dict['test_metrics']['bits']))

    ce = results_dict['calib_err']
    outfile.write(("{:>14.2f} &" * 4 + "\n").format(ce['gme'], ce['ece'], ce['mce'], ce['oce']))

    ood = results_dict['ood_tt']
    ent = results_dict['ood_ent']

    outfile.write(("{:>14.2f} &" * 5 + "\n").format(ent['geo_mean'],
                                                    ent['rot_rgb'],
                                                    ent['quickdraw'],
                                                    ent['noisy'],
                                                    ent['imagenet']))

    outfile.write((("{:>14.2f} &" * 5)[:-1] + "\\\\\n").format(100. * ood['geo_mean'],
                                                               100. * ood['rot_rgb'],
                                                               100. * ood['quickdraw'],
                                                               100. * ood['noisy'],
                                                               100. * ood['imagenet']))
    outfile.close()


def to_csv_row(results_dict, out_dir):
    pass

def to_raw(results_dict, out_dir):
    pass

def to_json(results_dict, out_dir):
    json.dump(results_dict, open(join(out_dir, 'results.json'), 'w'),
              sort_keys=True,  indent=2)

def to_console(results_dict, out_dir):

    logfile = open(join(out_dir, 'results.log'), 'w')
    ce = results_dict['calib_err']
    ece, mce, ovc, gme = ce['ece'], ce['mce'], ce['oce'], ce['gme']

    def log_write(line, endline='\n'):
        print("\t" + line, flush=True)
        logfile.write(line)
        logfile.write(endline)

    log_write('ACC     %.4f' % (results_dict['test_metrics']['acc']))
    log_write('BITS    %.4f' % (results_dict['test_metrics']['bits']))
    log_write('')

    log_write(('XCE     ' + '%-10s' * 3) % ('ECE', 'MCE', 'OVC'))
    log_write(('XCE     ' + '%-10.6f' * 3) % (ece, mce, ovc))
    log_write('XCE GM  %.6f' % (21.5443 * gme))
    log_write('')

    for i, test_type in enumerate(['ood_ent', 'ood_1t', 'ood_2t', 'ood_tt']):
        aucs = results_dict[test_type]
        labels_list = list(aucs.keys())

        if i == 0:
            log_write('DATASET  ' + ''.join(['%-16s' % (l) for l in labels_list]))
            mult = 1.
        else:
            mult = 100.

        log_write('%-7s  ' % (test_type.upper()) + ''.join(['%-16.4f' % (mult * aucs[l]) for l in labels_list]))

    logfile.close()
