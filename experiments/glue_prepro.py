import os
import argparse
import random
#path.append(os.getcwd())

import sys 
sys.path.insert(0,'/home/aakbar/mt_dnn_test/mt_dnn_master/') 


from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.glue.glue_utils import *

logger = create_logger(__name__, to_disk=True, log_file='glue_prepro.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--old_glue', action='store_true', help='whether it is old GLUE, refer official GLUE webpage for details')
    args = parser.parse_args()
    return args


def main(args):
    is_old_glue = args.old_glue
    root = args.root_dir
    os.chdir('/home/aakbar/mt_dnn_test/mt_dnn_master/') 
    assert os.path.exists(root)

    ######################################
    # Hate Classification tasks
    ######################################


    davidson_train_path = os.path.join(root, 'davidson/train.csv')
    davidson_dev_path = os.path.join(root, 'davidson/dev.csv')
    davidson_test_path = os.path.join(root, 'davidson/test.csv')

    hateval_train_path = os.path.join(root, 'hateval/train.csv')
    hateval_dev_path = os.path.join(root, 'hateval/dev.csv')
    hateval_test_path = os.path.join(root, 'hateval/test.csv')
    
    waseem_train_path = os.path.join(root, 'waseem/train.csv')
    waseem_dev_path = os.path.join(root, 'waseem/dev.csv')
    waseem_test_path = os.path.join(root, 'waseem/test.csv')

    founta_train_path = os.path.join(root, 'founta/train.csv')
    founta_dev_path = os.path.join(root, 'founta/dev.csv')
    founta_test_path = os.path.join(root, 'founta/test.csv')

    ######################################
    # Loading DATA
    ######################################

    davidson_train_data = load_davidson(davidson_train_path)
    davidson_dev_data = load_davidson(davidson_dev_path)
    davidson_test_data = load_davidson(davidson_test_path)
    logger.info('Loaded {} davidson train samples'.format(len(davidson_train_data)))
    logger.info('Loaded {} davidson dev samples'.format(len(davidson_dev_data)))
    logger.info('Loaded {} davidson test samples'.format(len(davidson_test_data)))

    hateval_train_data = load_hateval(hateval_train_path, header=True)
    hateval_dev_data = load_hateval(hateval_dev_path, header=True)
    hateval_test_data = load_hateval(hateval_test_path, is_train=True)
    logger.info('Loaded {} hateval train samples'.format(len(hateval_train_data)))
    logger.info('Loaded {} hateval dev samples'.format(len(hateval_dev_data)))
    logger.info('Loaded {} hateval test samples'.format(len(hateval_test_data)))
    
    waseem_train_data = load_waseem(waseem_train_path)
    waseem_dev_data = load_waseem(waseem_dev_path)
    waseem_test_data = load_waseem(waseem_test_path)
    logger.info('Loaded {} waseem train samples'.format(len(waseem_train_data)))
    logger.info('Loaded {} waseem dev samples'.format(len(waseem_dev_data)))
    logger.info('Loaded {} waseem test samples'.format(len(waseem_test_data)))

    founta_train_data = load_founta(founta_train_path)
    founta_dev_data = load_founta(founta_dev_path)
    founta_test_data = load_founta(founta_test_path)
    logger.info('Loaded {} founta train samples'.format(len(founta_train_data)))
    logger.info('Loaded {} founta dev samples'.format(len(founta_dev_data)))
    logger.info('Loaded {} founta test samples'.format(len(founta_test_data)))

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    # BUILD data
   
    davidson_train_fout = os.path.join(canonical_data_root, 'davidson_train.tsv')
    davidson_dev_fout = os.path.join(canonical_data_root, 'davidson_dev.tsv')
    davidson_test_fout = os.path.join(canonical_data_root, 'davidson_test.tsv')
    dump_rows(davidson_train_data, davidson_train_fout, DataFormat.PremiseOnly)
    dump_rows(davidson_dev_data, davidson_dev_fout, DataFormat.PremiseOnly)
    dump_rows(davidson_test_data, davidson_test_fout, DataFormat.PremiseOnly)
    logger.info('done with davidson')

    hateval_train_fout = os.path.join(canonical_data_root, 'hateval_train.tsv')
    hateval_dev_fout = os.path.join(canonical_data_root, 'hateval_dev.tsv')
    hateval_test_fout = os.path.join(canonical_data_root, 'hateval_test.tsv')
    dump_rows(hateval_train_data, hateval_train_fout, DataFormat.PremiseOnly)
    dump_rows(hateval_dev_data, hateval_dev_fout, DataFormat.PremiseOnly)
    dump_rows(hateval_test_data, hateval_test_fout, DataFormat.PremiseOnly)
    logger.info('done with hateval')

    waseem_train_fout = os.path.join(canonical_data_root, 'waseem_train.tsv')
    waseem_dev_fout = os.path.join(canonical_data_root, 'waseem_dev.tsv')
    waseem_test_fout = os.path.join(canonical_data_root, 'waseem_test.tsv')
    dump_rows(waseem_train_data, waseem_train_fout, DataFormat.PremiseOnly)
    dump_rows(waseem_dev_data, waseem_dev_fout, DataFormat.PremiseOnly)
    dump_rows(waseem_test_data, waseem_test_fout, DataFormat.PremiseOnly)
    logger.info('done with waseem')

    founta_train_fout = os.path.join(canonical_data_root, 'founta_train.tsv')
    founta_dev_fout = os.path.join(canonical_data_root, 'founta_dev.tsv')
    founta_test_fout = os.path.join(canonical_data_root, 'founta_test.tsv')
    dump_rows(founta_train_data, founta_train_fout, DataFormat.PremiseOnly)
    dump_rows(founta_dev_data, founta_dev_fout, DataFormat.PremiseOnly)
    dump_rows(founta_test_data, founta_test_fout, DataFormat.PremiseOnly)
    logger.info('done with founta')


if __name__ == '__main__':
    args = parse_args()
    main(args)
