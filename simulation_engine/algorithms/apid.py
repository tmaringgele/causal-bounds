import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from os.path import abspath, dirname
from types import SimpleNamespace
import torch
import numpy as np
from simulation_engine.algorithms.apid_src.src.models.apid import APID
from simulation_engine.util.alg_util import AlgUtil




class Apid:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # torch.set_default_dtype(torch.double)

    ALG_NAME = "apid"

    def bound_ATE(data, device='cpu'):
        query = "ATE"
        for idx, sim in data.iterrows():

            failed = False
            # try:
            bound_lower, bound_upper  = Apid.run_from_generate_data(sim, device)
            print(f"bound_ATE for idx {idx}: {bound_lower}, {bound_upper}")
            # except Exception as e:
            #     print(f"Error in {Apid.ALG_NAME}: {e}")
            #     failed = True
            #Flatten bounds to trivial ceils
            if failed | (bound_upper > AlgUtil.get_trivial_Ceils(query)[1]):
                bound_upper = AlgUtil.get_trivial_Ceils(query)[1] 
            if failed | (bound_lower < AlgUtil.get_trivial_Ceils(query)[0]): 
                bound_lower = AlgUtil.get_trivial_Ceils(query)[0]


            bounds_valid = bound_lower <= sim[query+'_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            
            data.at[idx, query+'_'+Apid.ALG_NAME+'_bound_lower'] = bound_lower
            data.at[idx, query+'_'+Apid.ALG_NAME+'_bound_upper'] = bound_upper
            data.at[idx, query+'_'+Apid.ALG_NAME+'_bound_valid'] = bounds_valid
            data.at[idx, query+'_'+Apid.ALG_NAME+'_bound_width'] = bounds_width
            data.at[idx, query+'_'+Apid.ALG_NAME+'_bound_failed'] = failed

        return data




    def run_from_generate_data(data, device='cpu'):
        # 1. Get synthetic IV-style data
        Y = data['Y']
        X = data['X']

        # Build treatment groups
        Y0 = torch.tensor(Y[X == 0], dtype=torch.float32).reshape(-1, 1)
        Y1 = torch.tensor(Y[X == 1], dtype=torch.float32).reshape(-1, 1)
        data_dict = {'Y0': Y0, 'Y1': Y1}

        # 2. Build APID config (with curvature regularization enabled)
        args = SimpleNamespace(
            model=SimpleNamespace(
                name='apid',
                dim_u=2,
                n_trans=15,
                tol=1e-4,
                aug_mode='s',
                n_quantiles=32,
                eps=0.5,
                batch_size=32,
                burn_in_epochs=10,
                q_epochs=10,
                curv_epochs=3,
                noise_std=0.001,
                lr=0.01,
                cf_only=True,
                ema_q=0.99,
                q_coeff=0.05,
                curv_coeff=0.2  # Enable curvature constraint!
            ),
            dataset=SimpleNamespace(name='synthetic_iv'),
            exp=SimpleNamespace(device=device, logging=False, seed=0, mlflow_uri=None)
        )

        # 3. Initialize model and select a factual input (for ECOU query)
        model = APID(args)
        idx = np.where((X == 0) & (np.abs(Y - 0.5) < 0.05))[0]
        if len(idx) == 0:
            print("No suitable factual units found with Y ≈ 0.5 and X = 0.")
            return

        i = idx[0]
        y_f = torch.tensor([[Y[i]]], dtype=torch.float32)
        t_f = int(X[i])
        f_dict = {'Y_f': y_f, 'T_f': t_f}

        # 4. Train APID
        model.fit(data_dict, f_dict, log=False)

        # 5. ECOU bounds for selected unit
        t_cf = 1 - t_f
        # cf_lb, cf_ub = model.get_bounds(
        #     factual_outcome=y_f,
        #     factual_treatment=t_f,
        #     counterfactual_treatment=t_cf,
        #     alpha=0.05,
        #     n_samples=500,
        #     mode='csm'
        # )

        # 6. ATE bounds across entire dataset
        ate_lb, ate_ub = Apid.get_ATE_bounds_from_model(model, X, Y, alpha=0.05, n_samples=500)

        print(f"Factual: A={t_f}, Y={y_f.item():.3f}")
        # print(f"Counterfactual ECOU bounds (A={t_cf}): [{cf_lb.item():.3f}, {cf_ub.item():.3f}]")
        print(f"APID ATE bounds: [{ate_lb:.3f}, {ate_ub:.3f}]")
        print(f"True ATE: {data['ATE_true']:.3f}, True PNS: {data['PNS_true']:.3f}")
        return ate_lb, ate_ub

    @staticmethod
    def get_ATE_bounds_from_model(model, X, Y, alpha=0.05, n_samples=500):
        """
        Compute ATE bounds by aggregating per-unit ECOU bounds.
        Enforces correct bound ordering to avoid inverted intervals.
        """
        lower_diffs = []
        upper_diffs = []

        for i, (x_i, y_i) in enumerate(zip(X, Y)):
            y_tensor = torch.tensor([[y_i]], dtype=torch.float32)
            x_i = int(x_i)

            # Compute and sort counterfactual bounds to ensure proper order
            lb_y1, ub_y1 = sorted(model.get_bounds(
                y_tensor,
                factual_treatment=x_i,
                counterfactual_treatment=1,
                alpha=alpha,
                n_samples=n_samples,
                mode='csm'
            ))
            lb_y0, ub_y0 = sorted(model.get_bounds(
                y_tensor,
                factual_treatment=x_i,
                counterfactual_treatment=0,
                alpha=alpha,
                n_samples=n_samples,
                mode='csm'
            ))

            # ATE_i bounds = worst-case subtraction
            lb_ate_i = lb_y1 - ub_y0
            ub_ate_i = ub_y1 - lb_y0

            # Final safety check
            if lb_ate_i > ub_ate_i:
                print(f"[Warning] Inverted ATE bounds at unit {i}: "
                    f"lb={lb_ate_i:.4f}, ub={ub_ate_i:.4f}. Fixing.")
                lb_ate_i, ub_ate_i = min(lb_ate_i, ub_ate_i), max(lb_ate_i, ub_ate_i)

            lower_diffs.append(lb_ate_i)
            upper_diffs.append(ub_ate_i)

        ATE_lower = torch.stack(lower_diffs).mean().item()
        ATE_upper = torch.stack(upper_diffs).mean().item()
        return ATE_lower, ATE_upper



    @staticmethod
    def run_direct():
        # 1. Define a fake dataset (replace with your own!)
        n = 1000
        Y0 = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=(n, 1)), dtype=torch.float32)
        Y1 = torch.tensor(np.random.normal(loc=1.5, scale=1.0, size=(n, 1)), dtype=torch.float32)
        data_dict = {'Y0': Y0, 'Y1': Y1}

        # 2. Create config manually (without Hydra)
        args = SimpleNamespace(
            model=SimpleNamespace(
                name='apid',
                dim_u=2,
                n_trans=15,
                tol=1e-4,
                aug_mode='s',
                n_quantiles=32,
                eps=0.5,
                batch_size=32,
                burn_in_epochs=10,
                q_epochs=10,
                curv_epochs=3,
                noise_std=0.001,
                lr=0.01,
                cf_only=True,
                ema_q=0.99,
                q_coeff=2.0,
                curv_coeff=0.0 #TODO: CHANGE 
            ),
            dataset=SimpleNamespace(name='custom'),
            exp=SimpleNamespace(device='cpu', logging=False, seed=0, mlflow_uri=None)
        )

        # 3. Instantiate and train model
        model = APID(args)
        f_dict = {'Y_f': torch.tensor([[0.0]]), 'T_f': 0}  # factual outcome Y=0.0, A=0
        model.fit(data_dict, f_dict, log=False)

        # 4. Get bounds
        cf_lb, cf_ub = model.get_bounds(
            factual_outcome=torch.tensor([[0.0]]),
            factual_treatment=0,
            counterfactual_treatment=1
        )
        print(f"ECOU bounds: [{cf_lb.item():.3f}, {cf_ub.item():.3f}]")

    @staticmethod
    @hydra.main(config_name=f'config.yaml', config_path='./apid_src/config/')
    def run(args: DictConfig):




        # Non-strict access to fields
        OmegaConf.set_struct(args, False)
        APID.logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

        # Initialisation of train_data_dict
        torch.set_default_device(args.exp.device)
        seed_everything(args.exp.seed)
        dataset = instantiate(args.dataset, _recursive_=True)

        data_dict = dataset.get_data()
        model = instantiate(args.model, args, _recursive_=True)

        f_dict = {'Y_f': torch.tensor([[args.dataset.Y_f]]), 'T_f': args.dataset.T_f}

        # try:
        model.fit(train_data_dict=data_dict, f_dict=f_dict, log=args.exp.logging)
        # except RuntimeError:
        #     pass

        model.mlflow_logger.experiment.set_terminated(model.mlflow_logger.run_id) if args.exp.logging else None
        # After training
        if hasattr(args.dataset, "Y_f"):
            bounds = []
            y_f_list = args.dataset.Y_f if isinstance(args.dataset.Y_f, list) else [args.dataset.Y_f]
            for y_f in y_f_list:
                y_f_tensor = torch.tensor([[y_f]], dtype=torch.float32).to(args.exp.device)

                # Assuming binary treatment: factual A = 0, counterfactual A = 1
                cf_lb, cf_ub = model.get_bounds(
                    factual_outcome=y_f_tensor,
                    factual_treatment=args.dataset.T_f,
                    counterfactual_treatment=1 - args.dataset.T_f
                )

                print(f"Counterfactual bounds for Y_f = {y_f}: [{cf_lb.item():.4f}, {cf_ub.item():.4f}]")
                bounds.append((y_f, cf_lb.item(), cf_ub.item()))

        # Save bounds to file
        print("\nHere come the ECOU bounds:")
        for y_f, lb, ub in bounds:
            print(f"{y_f}: [{lb:.3f}, {ub:.3f}]\n")
        # mlflow server --port=5000
        # $env:PYTHONPATH = "."
        # python runnables/train_apid.py +dataset=multi_modal +model=apid exp.seed=10 exp.logging=True exp.device=cpu
