import os
from copy import deepcopy
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace
import torch
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.compose import ColumnTransformer
from scipy.special import logit, expit
# from scipy.stats import qmc
from torch.quasirandom import SobolEngine
from .scripts import acquisition_functions, tune_input_warping

class PFNOptimizer(AbstractOptimizer):
    # Used for determining the version number of package used
    # primary_import = ""

    def __init__(self, api_config, pfn_file, minimize=True, acqf_optimizer_name="lbfgs", sobol_sampler=False,
                 device="cpu:0", fit_encoder_from_step=None, verbose=False, rand_bool=False, sample_only_valid=False,
                 round_suggests_to=4, min_initial_design=0, max_initial_design=None, rand_sugg_after_x_steps_of_stagnation=None,
                 fixed_initial_guess=None,minmax_encode_y=False,**acqf_kwargs):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        assert not 'fit_encoder' in acqf_kwargs
        AbstractOptimizer.__init__(self, api_config)
        # Do whatever other setup is needed
        # ...

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.device = device

        self.model = torch.load(pfn_file) if pfn_file.startswith('/') else torch.load(os.path.dirname(__file__) + '/' + pfn_file)

        self.X = []
        self.y = []
        self.api_config = {key: value for key, value in sorted(api_config.items())}
        self.hp_names = list(self.api_config.keys())
        # self.model.encoder.num_features = 18

        self.epsilon = 1e-8
        self.minimize = minimize
        self.sobol_sampler = sobol_sampler
        self.create_scaler()
        self.sobol = SobolEngine(len(self.max_values), scramble=True)
        self.acqf_optimizer_name = acqf_optimizer_name
        self.acqf_kwargs = acqf_kwargs
        self.fit_encoder_from_step = fit_encoder_from_step
        assert not (rand_bool and sample_only_valid)
        self.rand_bool = rand_bool
        self.sample_only_valid = sample_only_valid
        self.verbose = verbose
        self.round_suggests_to = round_suggests_to
        self.min_initial_design = min_initial_design
        self.max_initial_design = max_initial_design
        self.fixed_initial_guess = fixed_initial_guess
        self.minmax_encode_y = minmax_encode_y
        self.rand_sugg_after_x_steps_of_stagnation = rand_sugg_after_x_steps_of_stagnation
        self.model.eval()

        print(api_config)

    def create_scaler(self):

        list_of_scalers = []
        self.min_values = []
        self.max_values = []
        self.spaces = []
        self.types = []

        for i, feature in enumerate(self.api_config):
            # list_of_scalers.append((feature, MinMaxScaler(feature_range),i))
            self.spaces.append(self.api_config[feature].get("space", "bool"))
            self.types.append(self.api_config[feature]["type"])

            if self.types[-1] == "bool":
                feature_range = [0, 1]
            else:
                feature_range = list(self.api_config[feature]["range"])

            feature_range[0] = self.transform_feature(feature_range[0], -1)
            feature_range[1] = self.transform_feature(feature_range[1], -1)

            self.min_values.append(feature_range[0])
            self.max_values.append(feature_range[1])

        self.column_scaler = ColumnTransformer(list_of_scalers)
        self.max_values: np.array = np.array(self.max_values)
        self.min_values: np.array = np.array(self.min_values)

    def transform_feature_inverse(self, x, feature_index):

        if self.spaces[feature_index] == "log":
            x = np.exp(x)
        elif self.spaces[feature_index] == "logit":
            x = expit(x)
        if self.types[feature_index] == "int":
            if self.rand_bool:
                x = int(x) + int(np.random.rand() < (x-int(x)))
            else:
                x = int(np.round(x))
        elif self.types[feature_index] == "bool":
            if self.rand_bool:
                x = np.random.rand() < x
            else:
                x = bool(np.round(x))

        return x

    def transform_feature(self, x, feature_index):

        if np.isinf(x) or np.isnan(x):
            return 0

        if self.spaces[feature_index] == "log":
            x = np.log(x)

        elif self.spaces[feature_index] == "logit":
            x = logit(x)

        elif self.types[feature_index] == "bool":
            x = int(x)
        return x

    def random_suggest(self):
        self.rand_prev = True

        if self.sobol_sampler:

            # sampler = qmc.Sobol(d=len(self.max_values), scramble=False)
            # temp_guess = sampler.random_base2(m=len(self.max_values))
            temp_guess = self.sobol.draw(1).numpy()[0]
            temp_guess = temp_guess * (self.max_values - self.min_values) + self.min_values

            x_guess = {}
            for j, feature in enumerate(self.api_config):
                x = self.transform_feature_inverse(temp_guess[j], j)
                x_guess[feature] = x
            x_guess = [x_guess]

        else:
            x_guess = {}
            for i, feature in enumerate(self.api_config):
                temp_guess = np.random.uniform(self.min_values[i], self.max_values[i], 1)[0]
                temp_guess = self.transform_feature_inverse(temp_guess, i)

                x_guess[feature] = temp_guess
            x_guess = [x_guess]
        return x_guess

    def transform_back(self, x_guess):
        if self.round_suggests_to is not None:
            x_guess = np.round(x_guess, self.round_suggests_to)  # make sure very similar values are actually the same
        x_guess = x_guess * (self.max_values - self.min_values) + self.min_values
        x_guess = x_guess.tolist()
        return self.transform_inverse(x_guess)

    def min_max_encode(self, temp_X):
        # this, combined with transform is the inverse of transform_back
        temp_X = (temp_X - self.min_values) / (self.max_values - self.min_values)
        temp_X = torch.tensor(temp_X).to(torch.float32)
        temp_X = torch.clamp(temp_X, min=0., max=1.)
        return temp_X


    @torch.no_grad()
    def suggest(self, n_suggestions=1):
        """Get suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        assert n_suggestions == 1, "Only one suggestion at a time is supported"
        # Do whatever is needed to get the parallel guesses
        # ...
        # scaler = MinMaxScaler()
        # scaler.fit(self.X)
        try:
            num_initial_design = max(len(self.bounds), self.min_initial_design)
            if self.max_initial_design is not None:
                num_initial_design = min(num_initial_design, self.max_initial_design)
            if len(self.X) < num_initial_design:
                if len(self.X) == 0 and self.fixed_initial_guess is not None:
                    x_guess = [self.transform_back(np.array([self.fixed_initial_guess for _ in range(len(self.bounds))]))]
                else:
                    x_guess = self.random_suggest()
                return x_guess
            else:
                temp_X = np.array(self.X)
                temp_X = self.min_max_encode(temp_X)
                if self.minmax_encode_y:
                    temp_y = MinMaxScaler().fit_transform(np.array(self.y).reshape(-1, 1)).reshape(-1)
                else:
                    temp_y = np.array(self.y)
                temp_y = torch.tensor(temp_y).to(torch.float32)
                if self.rand_sugg_after_x_steps_of_stagnation is not None \
                        and len(self.y) > self.rand_sugg_after_x_steps_of_stagnation\
                        and not self.rand_prev:
                    if temp_y[:-self.rand_sugg_after_x_steps_of_stagnation].max() == temp_y.max():
                        print(f"Random suggestion after >= {self.rand_sugg_after_x_steps_of_stagnation} steps of stagnation")
                        x_guess = self.random_suggest()
                        return x_guess
                if self.verbose:
                    from matplotlib import pyplot as plt
                    print(f"{temp_X=}, {temp_y=}")
                    if temp_X.shape[1] == 2:
                        from scipy.stats import rankdata
                        plt.title('Observations, red -> blue.')
                        plt.scatter(temp_X[:,0], temp_X[:,1], cmap='RdBu', c=rankdata(temp_y))
                        plt.show()

                temp_X = temp_X.to(self.device)
                temp_y = temp_y.to(self.device)

                if self.fit_encoder_from_step and self.fit_encoder_from_step <= len(self.X):
                    with torch.enable_grad():
                        w = tune_input_warping.fit_input_warping(self.model, temp_X, temp_y)
                    temp_X_warped = w(temp_X).detach()
                else:
                    temp_X_warped = temp_X

                with torch.enable_grad():
                    if self.acqf_optimizer_name == "lbfgs":
                        def rand_sample_func(n):
                            pre_samples = torch.rand(n, temp_X_warped.shape[1], device='cpu')
                            back_transformed_samples = [self.transform_back(sample) for sample in pre_samples]
                            samples = np.array([self.transform(deepcopy(bt_sample)) for bt_sample in back_transformed_samples])
                            samples = self.min_max_encode(samples)
                            return samples.to(self.device)

                        if self.sample_only_valid:
                            rand_sample_func = rand_sample_func
                            # dims with bool or int are not continuous, thus no gradient opt is applied
                            dims_wo_gradient_opt = [i for i, t in enumerate(self.types) if t != "real"]
                        else:
                            rand_sample_func = None
                            dims_wo_gradient_opt = []

                        x_guess, x_options, eis, x_rs, x_rs_eis = acquisition_functions.optimize_acq_w_lbfgs(
                            self.model, temp_X_warped, temp_y, device=self.device,
                            verbose=self.verbose, rand_sample_func=rand_sample_func,
                            dims_wo_gradient_opt=dims_wo_gradient_opt, **{'apply_power_transform':True,**self.acqf_kwargs}
                        )

                    elif self.acqf_optimizer_name == 'adam':
                        x_guess = acquisition_functions.optimize_acq(self.model, temp_X_warped, temp_y, apply_power_transform=True, device=self.device, **self.acqf_kwargs
                                               ).detach().cpu().numpy()
                    else:
                        raise ValueError("Optimizer not recognized, set `acqf_optimizer_name` to 'lbfgs' or 'adam'")


                back_transformed_x_options = [self.transform_back(x) for x in x_options]
                opt_X = np.array([self.transform(deepcopy(transformed_x_options)) for transformed_x_options in back_transformed_x_options])
                opt_X = self.min_max_encode(opt_X)
                opt_new = ~(opt_X[:,None] == temp_X[None].cpu()).all(-1).any(1)
                for i, x in enumerate(opt_X):
                    if opt_new[i]:
                        if self.verbose: print(f"New point at pos {i}: {back_transformed_x_options[i], x_options[i]}")
                        self.rand_prev = False
                        return [back_transformed_x_options[i]]
                print('backup from initial rand search')
                back_transformed_x_options = [self.transform_back(x) for x in x_rs]
                opt_X = np.array([self.transform(deepcopy(transformed_x_options)) for transformed_x_options in back_transformed_x_options])
                opt_X = self.min_max_encode(opt_X)
                opt_new = ~(opt_X[:,None] == temp_X[None].cpu()).all(-1).any(1)
                for i, x in enumerate(opt_X):
                    if opt_new[i]:
                        if self.verbose: print(f"New point at pos {i}: {back_transformed_x_options[i], x_rs[i]} with ei {x_rs_eis[i]}")
                        self.rand_prev = False
                        return [back_transformed_x_options[i]]
                print("No new points found, random suggestion")
                return self.random_suggest()
        except Exception as e:
            raise e

    def transform(self, X_dict):
        X_tf = []
        for i, feature in enumerate(X_dict.keys()):
            X_dict[feature] = self.transform_feature(X_dict[feature], i)
            X_tf.append(X_dict[feature])
        return X_tf

    def transform_inverse(self, X_list):
        X_tf = {}
        for i, hp_name in enumerate(self.hp_names):
            X_tf[hp_name] = self.transform_feature_inverse(X_list[i], i)
        return X_tf


    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Update the model with new objective function observations
        # ...
        # No return statement needed
        if np.isinf(y) and y > 0:
            y[:] = 1e10

        if not np.isnan(y) and not np.isinf(y):
            assert len(y) == 1 and len(X) == 1, "Only one suggestion at a time is supported"
            X = {key: value for key, value in sorted(X[0].items())}
            assert list(X.keys()) == list(self.api_config.keys()) == list(self.hp_names) == list(
                self.space_x.param_list)
            if self.verbose:
                print(f"{X=}, {y=}")
            X = self.transform(X)
            if self.verbose:
                print(f"transformed {X=}")
            self.X.append(X)
            if self.minimize:
                self.y.append(-y[0])
            else:
                self.y.append(y[0])
        else:
            assert False

def test():
    from bayesmark.experiment import _build_test_problem, run_study, OBJECTIVE_NAMES
    #function_instance = _build_test_problem(model_name='ada', dataset='breast', scorer='nll', path=None)
    function_instance = _build_test_problem(model_name='kNN', dataset='boston', scorer='mse', path=None)

    # Setup optimizer
    api_config = function_instance.get_api_config()
    import os
    # check is file


    config = {
        "pfn_file":  'final_models/model_hebo_morebudget_9_unused_features_3.pt',
        "minimize": 1,
        "device": "cpu:0",
        "fit_encoder_from_step": None,
        'pre_sample_size': 1_000,
        'num_grad_steps': 15_000,
        'num_candidates': 10,
        'rand_bool': True,
    }
    opt = PFNOptimizer(api_config, verbose=True, **config)

    function_evals, timing, suggest_log = run_study(
        opt, function_instance, 50, 1, callback=None, n_obj=len(OBJECTIVE_NAMES),
    )


if __name__ == "__main__":
    import uuid
    from bayesmark.serialize import XRSerializer
    from bayesmark.cmd_parse import CmdArgs
    import bayesmark.cmd_parse as cmd
    import bayesmark.constants as cc

    description = "Run a study with one benchmark function and an optimizer"
    args = cmd.parse_args(cmd.experiment_parser(description))

    run_uuid = uuid.UUID(args[CmdArgs.uuid])


    # set global logging level
    logging.basicConfig(level=logging.DEBUG)
    # This is the entry point for experiments, so pass the class to experiment_main to use this optimizer.
    # This statement must be included in the wrapper class file:

    experiment_main(PFNOptimizer, args=args)