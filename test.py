import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils import init_project_mlflow, create_exp_mlflow
from parse_config import ConfigParser
import mlflow


def main(config):
    logger = config.get_logger('test')
    init_project_mlflow()
    run_name = create_exp_mlflow(model_name=config.resume)

    # setup data_loader instances
    try:
        data_loader = config.init_obj('data_loader_test', module_data)
        logger.info("Using test dataset")      
    except:
        data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
        )
        logger.warning("data_loader_test not set. Used validation dataset instead")

    # build model architecture
    model = config.init_obj('arch', module_arch, num_classes=len(data_loader.dataset.classes))
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    conf_matrix = torch.zeros(len(data_loader.dataset.classes),len(data_loader.dataset.classes)).to(device)

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            conf_matrix += module_metric.conf_matrix(output, target, device)

    n_samples = len(data_loader.sampler)
    log_metrics = {'loss': total_loss / n_samples}
    log_metrics.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log_metrics)
    
    ## config matrix ##
    try:
        config_matrix = config['conf_matrix']
        if config_matrix:
            module_metric.save_conf_matrix(conf_matrix=conf_matrix, 
                                        classes=data_loader.dataset.classes,
                                        saving_path=str(config.resume.parents[0]))
            logger.info(f"Confusion matrix saved at {str(config.resume.parents[0])}")
        else:
            logger.info("NO Confusion matrix saved")
    except:
        logger.info("NO Confusion matrix saved")    
    
    ## update mlflow server ##
    with mlflow.start_run(run_name=run_name) as run:
        params = {
        "epochs": config.config["trainer"]["epochs"],
        "arch": config.config["arch"]["type"],
        "batch_size": config.config["data_loader"]["args"]["batch_size"],
        "optimizer": config.config["optimizer"]["type"],
        "loss_name": config.config["loss"],
        "lr_scheduler": config.config["lr_scheduler"]["type"],
        }
        # # Log the parameters used for the model fit
        mlflow.log_params(params)

        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(log_metrics)

        # Log an instance of the trained model for later use
        mlflow.pytorch.log_model(model, "signal_model")  



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config_signals.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="saved/models/TestSignals/0119_081652_DepthSepWiseNeuralNetwork_50e/model_best.pth", type=str,
                      help='path to latest .pth checkpoint (default: None)')
    args.add_argument('-d', '--device', default="single", type=str,
                      help='indices of GPUs to enable (default: single)')
    args.add_argument('-o', '--output', default=None, type=str,
                      help='output folder name, timestamp if None')

    config = ConfigParser.from_args(args)
    main(config)
