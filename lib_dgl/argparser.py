from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
import yaml
from easydict import EasyDict as edict

"""Below is the annotation for all arguments used in this program:
    parser = argparse.ArgumentParser(description="Neighbor-Averaging-over-Relation-Subgraphs+SCR+GAMLP")
    # Model
    parser.add_argument("--method", type=str, default="SIGNV2")       # JK_GAMLP R_GAMLP SIGN SIGNV2
    parser.add_argument("--n-layers-1", type=int, default=3,
                        help="feature-transform: number of feed-forward layers")
    parser.add_argument("--n-layers-2", type=int, default=3,
                        help="feature-classification: number of feed-forward layers")
    parser.add_argument("--n-layers-3", type=int, default=3,
                        help="label-combining: number of feed-forward layers")
    parser.add_argument("--num-hidden", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="GCN - add alpha percent initial input h0 to transformed x")
    parser.add_argument("--input-drop", type=float, default=0.0,
                        help="input dropout of features after NARSsubset aggregator, before classification model")
    parser.add_argument("--att-drop", type=float, default=0.0,
                        help="attention dropout of model")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout for other unspecified modules")
    parser.add_argument("--label-drop", type=float, default=0.0,
                        help="label embedding dropout of model")
    parser.add_argument("--pre-process", default=True,
                        help="whether to process the input features in model")
    parser.add_argument("--residual", default=False,
                        help="whether to connect the input features before feature classification")
    parser.add_argument("--act", type=str, default="leaky_relu",
                        help="the activation function of the model")
    parser.add_argument("--pre-dropout", action='store_true', default=False,
                        help="JK_GAMLP pre-final label classification dropout")
    parser.add_argument("--bns", default=True, help="feedforwardNet para batchNormalization")
    # RLU - reliable label utilization
    parser.add_argument("--use-rlu", default=False,
                        help="whether to use the reliable data distillation")
    parser.add_argument("--label-num-hops",type=int,default=1, help="number of hops for label")
    parser.add_argument("--gamma", type=float, default=10,
                        help="parameter for the KL loss for reliable label utilization")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="the threshold for the node to be added into the model")
    # SCR - consistency regularization
    # Pseudo-Labelling
    parser.add_argument("--use_pseudo", action="store_false", default=True)
    parser.add_argument("--use_scr", action="store_false", default=True)
    parser.add_argument("--pseudo_warmup", type=int, default=10)
    parser.add_argument("--pseudo_lam", type=float, default=0.5)
    parser.add_argument("--pseudo_lam_max", type=float, default=0.7)
    parser.add_argument("--ramp_epochs", type=float, default=5)
    parser.add_argument("--tops", nargs='+', default=[0.95,0.90])
    parser.add_argument("--downs", nargs='+', default=[0.90,0.85])
    parser.add_argument("--ema_decay", type=float, default=0.8)
    parser.add_argument("--adap", action="store_false", default=True, 
                        help='if false, alpha == ema_decay, the smaller the closer teacher and student')
    parser.add_argument("--unsup_losstype", type=str, default='mse',
                        help="parameter for whic loss type to use")
    parser.add_argument("--sup_lam", type=float, default=1.0,
                        help='supervised loss factor')
    parser.add_argument("--unsup_lam", type=float, default=1.0,
                        help='scaling factor of consistency loss')
    parser.add_argument("--temp", type=float, default=1.0,
                        help="temperature of the output prediction, the smaller the sharper")
    parser.add_argument("--e_blocks", nargs='+', default=[10, 15, 25])
    parser.add_argument("--do_s", type=float, default=0.5)
    parser.add_argument("--do_w", type=float, default=0.2)
    # NARS
    parser.add_argument("--use-relation-subsets", type=str, default='lib_dgl/icdm2022_rand_subsets')
    parser.add_argument("--remake_subsets", default=False, help='if true, stored subsets will be replaced')
    parser.add_argument("--num-hops", type=int, default=6, help="number of hops")
    parser.add_argument("--cpu-preprocess", default=False, help="Preprocess on CPU")
    # dataset
    parser.add_argument("--dataset", type=str, default="icdm")
    # parser.add_argument("--session", default='session1', help="1 or 2")
    parser.add_argument("--cv-id", default=-1, help="which cv to load for train|val idx")
    parser.add_argument("--kfold", default=6, help="which cv to load for train|val idx")
    # traning
    parser.add_argument("--stages", nargs='+', default=[30], help="The epoch setting for each stage.")
    parser.add_argument("--fine_tune", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001, help='weight_decay')
    parser.add_argument("--class-weight", nargs='+', default=[1, 1.])
    parser.add_argument("--early_stopping", default=False)
    parser.add_argument("--stop-patience", type=int, default=1000)
    parser.add_argument("--loss_fcn", type=str, default='focal')
    parser.add_argument("--focal_gamma", type=int, default=2)
    parser.add_argument("--opt", type=str, default='adamw')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
    parser.add_argument('--nesterov', action='store_false', help='use nesterov')
    parser.add_argument('--sched', default='plateau')    # plateau cosine
    parser.add_argument('--warmup_epochs', default=3)    # cosine
    # parser.add_argument("--eval-every", type=int, default=2)
    # env
    parser.add_argument("--device-id", default='0')
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
"""



class ArgsParser(ArgumentParser):
    def __init__(self, description):
        super().__init__(formatter_class=RawDescriptionHelpFormatter, description=description)
        self.add_argument("-c","--config", help="configuration file to use")

    def _load_config(self, file_path):
        """
        Load config from yml/yaml file.
        Args:
            file_path (str): Path of the config file to be loaded.
        Returns: global config
        """
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"
        config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
        return config


    def parse_args(self, argv=None):
        _args = super(ArgsParser, self).parse_args(argv)
        assert _args.config is not None, \
            "Please specify --config=configure_file_path."
        args = self._load_config(_args.config)
        return edict(args)

if __name__=='__main__':
    args = ArgsParser(description="Neighbor-Averaging over Relation Subgraphs").parse_args()
    print(args)
    print(args.num_hidden)