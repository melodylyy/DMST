import argparse
from Arguments.run_args import get_run_args
from Arguments import testmodel_args, TimeXer_args, iTransformer_args, DLinear_args, Transformer_args, PDT_args, \
    SparseTSF_args, TimeDART_args, HPCC_args, GPHT_args, PatchMLP_args


def merge_args():
    model_args_dict = {
        'testmodel': testmodel_args,
        'GPHT': GPHT_args,
        'HPCC': HPCC_args,
        'TimeXer': TimeXer_args,
        'iTransformer': iTransformer_args,
        'DLinear': DLinear_args,
        'Transformer': Transformer_args,
        'PDT': PDT_args,
        'SparseTSF': SparseTSF_args,
        'TimeDART': TimeDART_args,
        'PatchMLP': PatchMLP_args,
    }

    # Run hyper-parameters
    parser1 = get_run_args()
    run_args = parser1.parse_args()

    # Model hyper-parameters
    model_args_file = model_args_dict[run_args.model]#
    parser2 = model_args_file.get_args()
    model_args = parser2.parse_args()

    # Merged hyper-parameters
    merged_parser = argparse.ArgumentParser(description='Merged parser')


    parser1_actions = parser1._actions
    parser2_actions = parser2._actions

    added_args = set()

    # Merge model hyper-parameters (model priority run)
    for action in parser2_actions:
        if action.dest == 'help':
            continue
        if action.dest not in added_args:
            merged_parser.add_argument(*action.option_strings,
                                       dest=action.dest,
                                       type=action.type,
                                       default=action.default,
                                       help=action.help,
                                       choices=action.choices)
            added_args.add(action.dest)

    # Merge run hyper-parameters
    for action in parser1_actions:
        if action.dest == 'help':
            continue
        if action.dest not in added_args:
            merged_parser.add_argument(*action.option_strings,
                                       dest=action.dest,
                                       type=action.type,
                                       default=action.default,
                                       help=action.help,
                                       choices=action.choices)
            added_args.add(action.dest)


    args = merged_parser.parse_args()

    return args, run_args, model_args
