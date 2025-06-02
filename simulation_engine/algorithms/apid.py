import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
from lightning_fabric.utilities.seed import seed_everything


class APID:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # torch.set_default_dtype(torch.double)


    @staticmethod
    @hydra.main(config_name=f'config.yaml', config_path='./apid/config/')
    def run(args: DictConfig):
        # Args:

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
